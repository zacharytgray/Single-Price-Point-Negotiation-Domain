"""
Prepare training data for HyperLoRA training.

Converts JSONL negotiation logs to Parquet tables with:
- trajectory_outcomes.parquet: One row per trajectory with outcome info
- decision_steps.parquet: One row per decision step for training

Usage:
    python -m src.data_prep.prepare_data --jsonl_path datasets/price_domain.jsonl --output_dir datasets/processed_tables
"""

import json
import collections
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load data from one or more JSONL files.
    
    Args:
        paths: List of paths to JSONL files
        
    Returns:
        List of parsed JSON objects
    """
    data = []
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Warning: {path} does not exist. Skipping.")
            continue
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError:
                    continue
    return data


def build_tables(
    paths: List[str], 
    k_history: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build trajectory outcome and decision step tables from JSONL data.
    
    Args:
        paths: List of paths to JSONL files
        k_history: Number of history items to include in each step
        
    Returns:
        Tuple of (trajectory_outcomes_df, decision_steps_df)
    """
    raw_data = load_jsonl(paths)
    print(f"Loaded {len(raw_data)} rows. Processing...")
    
    # Group by trajectory
    trajectories = collections.defaultdict(list)
    for row in raw_data:
        tid = row.get('trajectory_id')
        if tid:
            trajectories[tid].append(row)
            
    traj_outcomes_list = []
    decision_steps_list = []
    
    # Process each trajectory
    for tid, steps in trajectories.items():
        # Sort by turn
        def get_turn(r):
            if 'turn' in r: 
                return r['turn']
            if 'state' in r and 'timestep' in r['state']: 
                return r['state']['timestep']
            return -1
        
        steps.sort(key=get_turn)
        
        if not steps:
            continue
            
        first_step = steps[0]
        
        # Max turns
        max_turns = 20
        if 'state' in first_step and 'max_turns' in first_step['state']:
            max_turns = first_step['state']['max_turns']
            
        num_turns_observed = len(steps)
        
        # Check success
        has_accept = any(
            s.get('action', {}).get('type') == 'ACCEPT' 
            for s in steps
        )
        
        is_success = False
        is_failure = False
        
        if has_accept:
            is_success = True
        elif num_turns_observed == max_turns:
            is_failure = True
            
        if not is_success and not is_failure:
            # Trajectory is incomplete - Skip
            continue
            
        # Rho outcome
        rho_outcome = None
        if is_success:
            for s in steps:
                if s.get('rho') is not None:
                    rho_outcome = float(s['rho'])
                    break
        
        # Rho logged
        rho_logged = None
        for s in steps:
            if s.get('rho') is not None:
                rho_logged = float(s['rho'])
                break
                
        # Accepted Price
        accepted_price = None
        if is_success:
            # Try finding in ACCEPT row first
            for s in steps:
                if s.get('action', {}).get('type') == 'ACCEPT':
                    meta = s.get('meta', {})
                    if meta and 'accepted_price' in meta and meta['accepted_price'] is not None:
                        accepted_price = float(meta['accepted_price'])
                        break
            # Fallback to any row
            if accepted_price is None:
                for s in steps:
                    meta = s.get('meta', {})
                    if meta and 'accepted_price' in meta and meta['accepted_price'] is not None:
                        accepted_price = float(meta['accepted_price'])
                        break
                         
        traj_outcomes_list.append({
            'trajectory_id': tid,
            'max_turns': max_turns,
            'num_turns_observed': num_turns_observed,
            'success': is_success,
            'accepted_price': accepted_price,
            'rho_outcome': rho_outcome,
            'rho_logged': rho_logged
        })
        
        # Build Decision Steps
        running_history = []  # list of (role, price)
        
        rho_train = rho_outcome if is_success else -1.0
        
        for s in steps:
            curr_state = s.get('state', {})
            curr_action = s.get('action', {})
            
            turn = get_turn(s)
            role = curr_state.get('role', 'unknown')
            reservation_price = curr_state.get('reservation_price')
            
            # Price range
            price_range = curr_state.get('price_range')
            price_low = None
            price_high = None
            if isinstance(price_range, list) and len(price_range) >= 2:
                price_low = float(price_range[0])
                price_high = float(price_range[1])
                
            last_offer_price = curr_state.get('last_offer_price')
            
            # Get history slice
            hist_slice = running_history[-k_history:] if k_history > 0 else []
            history_roles = [str(x[0]) for x in hist_slice]
            history_prices = [float(x[1]) for x in hist_slice]
            
            # Target
            target_action = curr_action.get('type')
            target_price = curr_action.get('price')
            
            if target_action == 'ACCEPT':
                target_price = None
            elif target_price is not None:
                target_price = float(target_price)
                
            decision_steps_list.append({
                'trajectory_id': tid,
                'turn': turn,
                'role': role,
                'max_turns': max_turns,
                'turns_remaining': max_turns - turn,
                'reservation_price': float(reservation_price) if reservation_price is not None else None,
                'price_low': price_low,
                'price_high': price_high,
                'last_offer_price': float(last_offer_price) if last_offer_price is not None else None,
                'history_roles': history_roles,
                'history_prices': history_prices,
                'target_action': target_action,
                'target_price': target_price,
                'success': is_success,
                'rho_train': float(rho_train) if rho_train is not None else -1.0
            })
            
            # Update history AFTER recording step
            if target_action == 'OFFER' and target_price is not None:
                running_history.append((role, float(target_price)))
                
    df_outcomes = pd.DataFrame(traj_outcomes_list)
    df_steps = pd.DataFrame(decision_steps_list)
    return df_outcomes, df_steps


def validate(df_outcomes: pd.DataFrame, df_steps: pd.DataFrame) -> bool:
    """
    Validate the generated tables for consistency.
    
    Args:
        df_outcomes: Trajectory outcomes DataFrame
        df_steps: Decision steps DataFrame
        
    Returns:
        True if all validations pass
    """
    print("\n--- Validation Report ---")
    print(f"Total Trajectories: {len(df_outcomes)}")
    
    all_passed = True
    
    if not df_outcomes.empty:
        success_count = df_outcomes['success'].sum()
        fail_count = len(df_outcomes) - success_count
        print(f"Successes: {success_count}")
        print(f"Failures: {fail_count}")
        
    print(f"Total Decision Steps: {len(df_steps)}")
    
    if df_steps.empty:
        print("Warning: No decision steps generated.")
        return False

    # Check rho_train consistency within trajectories
    inconsistent = df_steps.groupby('trajectory_id')['rho_train'].nunique()
    if (inconsistent > 1).any():
        print("FAILED: Found trajectories with varying rho_train values per step!")
        bad_ids = inconsistent[inconsistent > 1].index.tolist()
        print(f"Sample inconsistent IDs: {bad_ids[:3]}")
        all_passed = False
    else:
        print("PASSED: rho_train is consistent within each trajectory.")
        
    # Check failures have rho_train == -1.0
    failures_steps = df_steps[~df_steps['success']]
    if not failures_steps.empty:
        bad_failures = failures_steps[failures_steps['rho_train'] != -1.0]
        if not bad_failures.empty:
            print(f"FAILED: Found {len(bad_failures)} failure steps with rho_train != -1.0")
            all_passed = False
        else:
            print("PASSED: All failure steps have rho_train == -1.0")
    else:
        print("INFO: No failure steps to validate.")
        
    # Check schema / nulls
    print("\nNull Value Checks:")
    print(df_steps[['rho_train', 'target_action']].isnull().sum())
    
    # Check history structure
    sample_roles = df_steps['history_roles'].iloc[0] if not df_steps.empty else []
    sample_prices = df_steps['history_prices'].iloc[0] if not df_steps.empty else []
    print(f"\nSample History Roles: {sample_roles}")
    print(f"Sample History Prices: {sample_prices}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL negotiation data to Parquet tables"
    )
    parser.add_argument(
        "--jsonl_path", 
        type=str, 
        default="datasets/price_domain.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="datasets/processed_tables",
        help="Directory for output Parquet files"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=8,
        help="Number of history items to include per step"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Build tables
    outcomes, steps = build_tables([args.jsonl_path], k_history=args.k)
    
    # Save
    outcomes.to_parquet(out_path / "trajectory_outcomes.parquet", index=False)
    steps.to_parquet(out_path / "decision_steps.parquet", index=False)
    
    print(f"\nTables written to {out_path}")
    
    # Validate
    validate(outcomes, steps)


if __name__ == "__main__":
    main()
