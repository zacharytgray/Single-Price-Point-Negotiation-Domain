"""
Dataset Writer for offline RL training data.

Logs negotiation episodes to JSONL format for use with
Decision Transformers, offline RL, and HyperLoRA training.
"""

import os
import json
import dataclasses
from typing import List, Dict, Any, Optional

from src.core.price_structures import PriceAction, PriceState


class DatasetWriter:
    """
    Handles logging negotiation episodes to a JSONL file for offline RL/Decision Transformer training.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the dataset writer.
        
        Args:
            filepath: Path to the JSONL output file
        """
        self.filepath = filepath
        
        # Ensure directory exists
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        self.episode_buffer: List[Dict] = []
        
    def add_step(
        self, 
        state: PriceState, 
        action: PriceAction, 
        reward: float,
        agent_metadata: Dict,
        meta: Dict,
        trajectory_id: str,
        terminal: bool = False
    ) -> None:
        """
        Buffer a single step from the negotiation.
        
        Args:
            state: Current price state
            action: Action taken
            reward: Immediate reward
            agent_metadata: Agent information (strategy, params)
            meta: Episode metadata (ZOPA, prices)
            trajectory_id: Unique ID for this trajectory
            terminal: Whether this is the final step
        """
        # Helper for float rounding
        def round_floats(obj):
            if isinstance(obj, float):
                return round(obj, 2)
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [round_floats(x) for x in obj]
            return obj

        step_record = {
            "trajectory_id": trajectory_id,
            "t": state.timestep,
            "agent": round_floats(agent_metadata),
            "state": round_floats(state.to_dict()),
            "action": round_floats(dataclasses.asdict(action)),
            "reward": round(reward, 4),
            "terminal": terminal,
            "meta": round_floats(meta)
        }
        self.episode_buffer.append(step_record)
        
    def flush_episode(
        self, 
        outcome_meta: Dict, 
        agent_rewards: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Post-process the current episode buffer to calculate Rho and write to disk.
        
        Rho is calculated as: 
            rho = (Final Price - Seller Min) / (Buyer Max - Seller Min)
        
        Same rho value is assigned to every step in the trajectory.
        
        Args:
            outcome_meta: Outcome information (agreement, price, turns)
            agent_rewards: Optional per-agent final rewards
        """
        if not self.episode_buffer:
            return

        # Check agreement and get final price
        if not outcome_meta.get("agreement", False):
            # No agreement => rho undefined => skip writing
            self.episode_buffer = [] 
            return

        final_price = outcome_meta.get("price")
        if final_price is None:
            self.episode_buffer = []
            return

        # Extract trajectory constants from the first step's metadata
        first_step = self.episode_buffer[0]
        meta_start = first_step["meta"]
        
        min_acceptable = meta_start.get("zopa_low")     # Seller Min
        max_acceptable = meta_start.get("zopa_high")    # Buyer Max

        if min_acceptable is None or max_acceptable is None:
            self.episode_buffer = []
            return

        # Calculate Rho
        denominator = max_acceptable - min_acceptable
        rho = 0.0
        if denominator != 0:
            rho = (final_price - min_acceptable) / denominator
        
        rho = round(rho, 4)

        # Write records with cleaned structure
        with open(self.filepath, 'a', encoding='utf-8') as f:
            for step in self.episode_buffer:
                
                # Map State to match template
                raw_state = step["state"]
                clean_state = {
                    "timestep": raw_state.get("timestep"),
                    "max_turns": raw_state.get("max_turns"),
                    "role": raw_state.get("role"),
                    "last_offer_price": raw_state.get("last_offer_price"),
                    "offer_history": raw_state.get("offer_history"),
                    "reservation_price": raw_state.get("true_reservation_price"),
                    "price_range": raw_state.get("public_price_range")
                }

                # Map Meta to match template
                clean_outcome = {
                    "agreement_bool": outcome_meta.get("agreement"),
                    "price": outcome_meta.get("price"),
                    "num_turns": outcome_meta.get("turns")
                }

                raw_meta = step["meta"]
                clean_meta = {
                    "buyer_max": raw_meta.get("buyer_max"),
                    "seller_min": raw_meta.get("seller_min"),
                    "zopa_low": raw_meta.get("zopa_low"),
                    "zopa_high": raw_meta.get("zopa_high"),
                    "zopa_width": raw_meta.get("zopa_width"),
                    "accepted_price": raw_meta.get("accepted_price"),
                    "episode_outcome": clean_outcome
                }

                # Map Agent to match template
                raw_agent = step["agent"]
                clean_agent = {
                    "strategy": raw_agent.get("strategy"),
                    "strategy_params": raw_agent.get("strategy_params")
                }

                # Assemble final record
                record = {
                    "trajectory_id": step["trajectory_id"],
                    "turn": step["t"],
                    "agent": clean_agent,
                    "state": clean_state,
                    "action": step["action"],
                    "is_terminal": step["terminal"],
                    "meta": clean_meta,
                    "rho": rho
                }
                
                f.write(json.dumps(record) + "\n")
        
        self.episode_buffer = []
        
    def flush_episode_with_failure(self, outcome_meta: Dict) -> None:
        """
        Write the episode buffer with rho=-1.0 for failed negotiations.
        
        This is used when agreement was not reached but we still want
        to include the trajectory for training purposes.
        
        Args:
            outcome_meta: Outcome information
        """
        if not self.episode_buffer:
            return
            
        first_step = self.episode_buffer[0]
        meta_start = first_step["meta"]
        
        rho = -1.0  # Failure marker
        
        with open(self.filepath, 'a', encoding='utf-8') as f:
            for step in self.episode_buffer:
                raw_state = step["state"]
                clean_state = {
                    "timestep": raw_state.get("timestep"),
                    "max_turns": raw_state.get("max_turns"),
                    "role": raw_state.get("role"),
                    "last_offer_price": raw_state.get("last_offer_price"),
                    "offer_history": raw_state.get("offer_history"),
                    "reservation_price": raw_state.get("true_reservation_price"),
                    "price_range": raw_state.get("public_price_range")
                }

                clean_outcome = {
                    "agreement_bool": outcome_meta.get("agreement", False),
                    "price": outcome_meta.get("price"),
                    "num_turns": outcome_meta.get("turns")
                }

                raw_meta = step["meta"]
                clean_meta = {
                    "buyer_max": raw_meta.get("buyer_max"),
                    "seller_min": raw_meta.get("seller_min"),
                    "zopa_low": raw_meta.get("zopa_low"),
                    "zopa_high": raw_meta.get("zopa_high"),
                    "zopa_width": raw_meta.get("zopa_width"),
                    "accepted_price": None,
                    "episode_outcome": clean_outcome
                }

                raw_agent = step["agent"]
                clean_agent = {
                    "strategy": raw_agent.get("strategy"),
                    "strategy_params": raw_agent.get("strategy_params")
                }

                record = {
                    "trajectory_id": step["trajectory_id"],
                    "turn": step["t"],
                    "agent": clean_agent,
                    "state": clean_state,
                    "action": step["action"],
                    "is_terminal": step["terminal"],
                    "meta": clean_meta,
                    "rho": rho
                }
                
                f.write(json.dumps(record) + "\n")
        
        self.episode_buffer = []
