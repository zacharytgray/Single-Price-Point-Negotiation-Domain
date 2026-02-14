"""
Generate Concession Curves from Janus Benchmark Results.

This script analyzes the CSV output from run_full_janus_benchmark.py and generates
concession curve plots showing how offers evolve over time for each strategy.

For each strategy, it creates plots showing:
- Janus as Seller vs Opponent (Buyer)
- Janus as Buyer vs Opponent (Seller)

Each plot shows normalized offer prices over turns, with individual episode traces
(faint) and average curves (bold).

Usage:
    python generate_concession_curves.py --csv logs/janus_full_benchmark_20260213_173731.csv
    python generate_concession_curves.py --csv logs/janus_full_benchmark_*.csv --output_dir concession_plots
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


def compute_average_curve(curves: List[List[Tuple[int, float]]], max_turns: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average curve from multiple episode curves.
    
    Returns:
        (turns, avg_values) arrays
    """
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
    prefix: str = ""
):
    """
    Create a concession curve plot for a specific strategy and Janus role.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Determine colors and labels based on Janus role
    if janus_role == 'seller':
        janus_color = '#e74c3c'  # Red for seller (decreasing)
        opponent_color = '#3498db'  # Blue for buyer (increasing)
        janus_label = 'Janus (Seller)'
        opponent_label = 'Opponent (Buyer)'
        title_suffix = 'Janus as Seller vs Opponent (Buyer)'
    else:
        janus_color = '#3498db'  # Blue for buyer (increasing)
        opponent_color = '#e74c3c'  # Red for seller (decreasing)
        janus_label = 'Janus (Buyer)'
        opponent_label = 'Opponent (Seller)'
        title_suffix = 'Janus as Buyer vs Opponent (Seller)'
    
    # Plot individual episode curves (faint)
    for curve in janus_curves:
        turns, values = zip(*curve)
        ax.plot(turns, values, color=janus_color, alpha=0.15, linewidth=0.8)
    
    for curve in opponent_curves:
        turns, values = zip(*curve)
        ax.plot(turns, values, color=opponent_color, alpha=0.15, linewidth=0.8)
    
    # Plot average curves (bold)
    if janus_curves:
        avg_turns, avg_values = compute_average_curve(janus_curves)
        if len(avg_turns) > 0:
            ax.plot(avg_turns, avg_values, color=janus_color, linewidth=3, 
                   label=f'{janus_label} (avg, n={len(janus_curves)})', marker='o', markersize=4)
    
    if opponent_curves:
        avg_turns, avg_values = compute_average_curve(opponent_curves)
        if len(avg_turns) > 0:
            ax.plot(avg_turns, avg_values, color=opponent_color, linewidth=3,
                   label=f'{opponent_label} (avg, n={len(opponent_curves)})', marker='s', markersize=4)
    
    # Add fair split line
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Fair Split (50%)')
    
    # Formatting
    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Offer (0=Seller Res, 1=Buyer Res)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concession Curves: {strategy}\n{title_suffix}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 21)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f"Episodes: {len(janus_curves)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
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
    prefix: str = ""
):
    """
    Create a comparison plot showing average concession curves for multiple strategies.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, janus_role)
        
        # Plot only Janus average curves for comparison
        if janus_curves:
            avg_turns, avg_values = compute_average_curve(janus_curves)
            if len(avg_turns) > 0:
                linestyle = '-' if janus_role == 'seller' else '--'
                ax.plot(avg_turns, avg_values, color=colors[i], linewidth=2.5,
                       label=f'{strategy}', linestyle=linestyle, marker='o', markersize=3)
    
    # Add fair split line
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1.5, label='Fair Split')
    
    # Formatting
    role_title = 'Janus as Seller' if janus_role == 'seller' else 'Janus as Buyer'
    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Offer (0=Seller Res, 1=Buyer Res)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concession Curve Comparison: {role_title}\nJanus Average Curves Across Strategies', 
                fontsize=13, fontweight='bold')
    ax.set_xlim(0, 21)
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
        default='concession_plots',
        help='Directory to save plots (default: concession_plots)'
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
    
    args = parser.parse_args()
    
    # Expand wildcards
    csv_files = glob.glob(args.csv)
    if not csv_files:
        print(f"{Fore.RED}No files found matching: {args.csv}{Fore.RESET}")
        sys.exit(1)
    
    # Use the most recent file if multiple match
    csv_path = sorted(csv_files)[-1]
    print(f"{Fore.CYAN}Using: {csv_path}{Fore.RESET}")
    
    # Load data
    df = load_results(csv_path)
    
    # Filter out micro strategies
    df = df[~df['strategy'].str.startswith('micro_')].copy()
    
    # Get unique strategies
    strategies = sorted(df['strategy'].unique())
    print(f"\nFound {len(strategies)} strategies: {', '.join(strategies)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots for each strategy
    print(f"\n{Fore.CYAN}Generating concession curves...{Fore.RESET}")
    
    for strategy in strategies:
        print(f"\n  Processing: {strategy}")
        
        # Janus as Seller
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, 'seller')
        if janus_curves or opponent_curves:
            plot_concession_curves(
                janus_curves, opponent_curves, strategy, 'seller',
                args.output_dir, args.prefix
            )
        else:
            print(f"    No data for Janus as Seller")
        
        # Janus as Buyer
        janus_curves, opponent_curves = extract_concession_curves(df, strategy, 'buyer')
        if janus_curves or opponent_curves:
            plot_concession_curves(
                janus_curves, opponent_curves, strategy, 'buyer',
                args.output_dir, args.prefix
            )
        else:
            print(f"    No data for Janus as Buyer")
    
    # Generate comparison plots if requested
    if args.comparison:
        print(f"\n{Fore.CYAN}Generating comparison plots...{Fore.RESET}")
        create_strategy_comparison_plot(df, strategies, 'seller', args.output_dir, args.prefix)
        create_strategy_comparison_plot(df, strategies, 'buyer', args.output_dir, args.prefix)
    
    print(f"\n{Fore.GREEN}All plots saved to: {args.output_dir}{Fore.RESET}")


if __name__ == '__main__':
    main()
