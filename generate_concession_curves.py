"""
Generate Concession Curves from Janus Benchmark Results.

This script analyzes the CSV output from run_full_janus_benchmark.py and generates
concession curve plots showing how offers evolve over time for each strategy.

One concession curve plot is generated per violin plot produced by analyze_benchmark_results.py:
- One plot per (strategy, role) pair: "Benchmarked Model as Seller" and "Benchmarked Model as Buyer"

Each plot shows:
- Faint lines for every individual episode's offer sequence (both model and opponent)
- Bold average curves on top (one per agent, color-coded)
- Fair split reference line at 0.5

Usage:
    python generate_concession_curves.py --csv logs/janus_full_benchmark_20260213_173731.csv
    python generate_concession_curves.py --csv logs/janus_full_benchmark_*.csv --output_dir concession_plots
    python generate_concession_curves.py --csv logs/base_*.csv --output_dir concession_plots --model_name "Base Qwen2-7B"
"""

import argparse
import os
import sys
import glob
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agents.price_strategies import EXCLUDED_FROM_BENCHMARK

init(autoreset=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    return df


def parse_offer_history(offer_history_json: str) -> List[Dict]:
    """Parse the offer history JSON string."""
    try:
        return json.loads(offer_history_json)
    except json.JSONDecodeError:
        return []


def normalize_offer(price: float, seller_min: float, buyer_max: float) -> float:
    """
    Normalize offer price to [0, 1] range.
    0 = seller's reservation price (seller_min)
    1 = buyer's reservation price (buyer_max)
    """
    zopa_width = buyer_max - seller_min
    if zopa_width == 0:
        return 0.5
    return (price - seller_min) / zopa_width


def extract_concession_curves(
    df: pd.DataFrame,
    strategy: str,
    janus_role: str
) -> Tuple[List[List[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    """
    Extract concession curves for a specific strategy and Janus role.
    
    Returns:
        (janus_curves, opponent_curves) where each is a list of episodes,
        and each episode is a list of (turn, normalized_price) tuples.
    """
    # Filter for the specific strategy and Janus role
    filtered = df[(df['strategy'] == strategy) & (df['janus_role'] == janus_role)]
    
    janus_curves = []
    opponent_curves = []
    
    for _, row in filtered.iterrows():
        offer_history = parse_offer_history(row['offer_history'])
        if not offer_history:
            continue
        
        seller_min = row['seller_min']
        buyer_max = row['buyer_max']
        
        janus_episode = []
        opponent_episode = []
        
        for event in offer_history:
            if event['action'] == 'OFFER':
                turn = event['turn']
                price = event['price']
                normalized = normalize_offer(price, seller_min, buyer_max)
                
                if event['agent'] == 'janus':
                    janus_episode.append((turn, normalized))
                else:
                    opponent_episode.append((turn, normalized))
        
        if janus_episode:
            janus_curves.append(janus_episode)
        if opponent_episode:
            opponent_curves.append(opponent_episode)
    
    return janus_curves, opponent_curves


def get_max_turn(curves: List[List[Tuple[int, float]]]) -> int:
    """Get the maximum turn number across all episode curves."""
    max_t = 0
    for curve in curves:
        for turn, _ in curve:
            if turn > max_t:
                max_t = turn
    return max_t if max_t > 0 else 20


def compute_average_curve(curves: List[List[Tuple[int, float]]], max_turns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average curve from multiple episode curves.
    
    Returns:
        (turns, avg_values) arrays
    """
    if max_turns is None:
        max_turns = get_max_turn(curves)

    # Create a matrix: rows = episodes, cols = turns
    values_by_turn = {t: [] for t in range(1, max_turns + 1)}
    
    for curve in curves:
        for turn, value in curve:
            if 1 <= turn <= max_turns:
                values_by_turn[turn].append(value)
    
    turns = []
    avg_values = []
    
    for t in range(1, max_turns + 1):
        if values_by_turn[t]:
            turns.append(t)
            avg_values.append(np.mean(values_by_turn[t]))
    
    return np.array(turns), np.array(avg_values)


def plot_concession_curves(
    janus_curves: List[List[Tuple[int, float]]],
    opponent_curves: List[List[Tuple[int, float]]],
    strategy: str,
    janus_role: str,
    output_dir: str,
    prefix: str = "",
    model_name: str = "Janus"
):
    """
    Create a concession curve plot for a specific strategy and Janus role.

    Mirrors one violin plot from analyze_benchmark_results.py:
    - One plot per (strategy, role) pair
    - Faint lines = individual episode offer sequences for both agents
    - Bold lines = average concession curve per agent
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine colors and labels based on role
    if janus_role == 'seller':
        model_color = '#e74c3c'   # Red for seller (offers decrease toward seller_min)
        opponent_color = '#3498db'  # Blue for buyer (offers increase toward buyer_max)
        model_label = f'{model_name} (Seller)'
        opponent_label = 'Opponent (Buyer)'
        title_suffix = f'{model_name} as Seller vs {strategy} (Buyer opponent)'
    else:
        model_color = '#3498db'   # Blue for buyer
        opponent_color = '#e74c3c'  # Red for seller
        model_label = f'{model_name} (Buyer)'
        opponent_label = 'Opponent (Seller)'
        title_suffix = f'{model_name} as Buyer vs {strategy} (Seller opponent)'

    # Determine dynamic x-axis limit from actual data
    all_curves = janus_curves + opponent_curves
    max_turn = get_max_turn(all_curves) if all_curves else 20

    # Plot individual episode curves (faint) — all rounds contribute
    for curve in janus_curves:
        turns, values = zip(*curve)
        ax.plot(turns, values, color=model_color, alpha=0.15, linewidth=0.8)

    for curve in opponent_curves:
        turns, values = zip(*curve)
        ax.plot(turns, values, color=opponent_color, alpha=0.15, linewidth=0.8)

    # Plot average curves (bold) on top of faint lines
    if janus_curves:
        avg_turns, avg_values = compute_average_curve(janus_curves, max_turns=max_turn)
        if len(avg_turns) > 0:
            ax.plot(avg_turns, avg_values, color=model_color, linewidth=3,
                   label=f'{model_label} (avg, n={len(janus_curves)})', marker='o', markersize=4)

    if opponent_curves:
        avg_turns, avg_values = compute_average_curve(opponent_curves, max_turns=max_turn)
        if len(avg_turns) > 0:
            ax.plot(avg_turns, avg_values, color=opponent_color, linewidth=3,
                   label=f'{opponent_label} (avg, n={len(opponent_curves)})', marker='s', markersize=4)

    # Fair split reference line
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Fair Split (50%)')

    # Formatting
    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Offer (0=Seller Res, 1=Buyer Res)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concession Curves: {strategy}\n{title_suffix}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max_turn + 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Stats box: show episode counts for both agents
    n_model = len(janus_curves)
    n_opp = len(opponent_curves)
    stats_text = f"{model_label}: {n_model} episodes\n{opponent_label}: {n_opp} episodes"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot — filename mirrors violin plot naming convention
    safe_strategy = strategy.replace(' ', '_').replace('/', '_')
    filename = f"{prefix}{safe_strategy}_{janus_role}_concession.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {filepath}")


def create_strategy_comparison_plot(
    df: pd.DataFrame,
    strategies: List[str],
    janus_role: str,
    output_dir: str,
    prefix: str = "",
    model_name: str = "Janus"
):
    """
    Create a comparison plot showing average concession curves for multiple strategies.
    Only the benchmarked model's average curve is shown per strategy (for cross-strategy comparison).
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    global_max_turn = 0

    for i, strategy in enumerate(strategies):
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, janus_role)

        if janus_curves:
            all_curves = janus_curves + opponent_curves
            max_turn = get_max_turn(all_curves)
            global_max_turn = max(global_max_turn, max_turn)

            avg_turns, avg_values = compute_average_curve(janus_curves, max_turns=max_turn)
            if len(avg_turns) > 0:
                linestyle = '-' if janus_role == 'seller' else '--'
                ax.plot(avg_turns, avg_values, color=colors[i], linewidth=2.5,
                       label=f'{strategy} (n={len(janus_curves)})', linestyle=linestyle, marker='o', markersize=3)

    if global_max_turn == 0:
        global_max_turn = 20

    # Fair split reference line
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1.5, label='Fair Split')

    # Formatting
    role_title = f'{model_name} as Seller' if janus_role == 'seller' else f'{model_name} as Buyer'
    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Offer (0=Seller Res, 1=Buyer Res)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concession Curve Comparison: {role_title}\n{model_name} Average Curves Across Strategies',
                fontsize=13, fontweight='bold')
    ax.set_xlim(0, global_max_turn + 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    filename = f"{prefix}comparison_{janus_role}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved comparison: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate concession curves from Janus benchmark results'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to benchmark CSV file (supports wildcards)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save plots (default: concession_curves subfolder in CSV directory)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Also generate strategy comparison plots'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Janus',
        help='Display name for the benchmarked model (default: Janus)'
    )

    args = parser.parse_args()
    
    # Expand wildcards
    csv_files = glob.glob(args.csv)
    if not csv_files:
        print(f"{Fore.RED}No files found matching: {args.csv}{Fore.RESET}")
        sys.exit(1)
    
    # Use the most recent file if multiple match
    csv_path = sorted(csv_files)[-1]
    print(f"{Fore.CYAN}Using: {csv_path}{Fore.RESET}")
    
    # Set default output directory to concession_curves subfolder in CSV directory
    if args.output_dir is None:
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        args.output_dir = os.path.join(csv_dir, "concession_curves")
    
    # Load data
    df = load_results(csv_path)
    
    # Filter out strategies excluded from benchmarking (e.g. random oracle)
    df = df[~df['strategy'].isin(EXCLUDED_FROM_BENCHMARK)].copy()
    
    # Get unique strategies
    strategies = sorted(df['strategy'].unique())
    print(f"\nFound {len(strategies)} strategies: {', '.join(strategies)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots for each strategy
    print(f"\n{Fore.CYAN}Generating concession curves...{Fore.RESET}")
    
    for strategy in strategies:
        print(f"\n  Processing: {strategy}")

        # Model as Seller (mirrors violin: "Model as Seller vs {strategy} (Buyer opponent)")
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, 'seller')
        if janus_curves or opponent_curves:
            plot_concession_curves(
                janus_curves, opponent_curves, strategy, 'seller',
                args.output_dir, args.prefix, model_name=args.model_name
            )
        else:
            print(f"    No data for {args.model_name} as Seller")

        # Model as Buyer (mirrors violin: "Model as Buyer vs {strategy} (Seller opponent)")
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, 'buyer')
        if janus_curves or opponent_curves:
            plot_concession_curves(
                janus_curves, opponent_curves, strategy, 'buyer',
                args.output_dir, args.prefix, model_name=args.model_name
            )
        else:
            print(f"    No data for {args.model_name} as Buyer")

    # Generate cross-strategy comparison plots if requested
    if args.comparison:
        print(f"\n{Fore.CYAN}Generating comparison plots...{Fore.RESET}")
        create_strategy_comparison_plot(df, strategies, 'seller', args.output_dir, args.prefix, model_name=args.model_name)
        create_strategy_comparison_plot(df, strategies, 'buyer', args.output_dir, args.prefix, model_name=args.model_name)
    
    print(f"\n{Fore.GREEN}All plots saved to: {args.output_dir}{Fore.RESET}")


if __name__ == '__main__':
    main()
