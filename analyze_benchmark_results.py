"""
Analyze Janus Benchmark Results - Generate Violin Plots for Utility Comparison.

This script analyzes the CSV output from run_full_janus_benchmark.py and generates
violin plots comparing Janus vs opponent utilities for each strategy, split by
Janus role (buyer vs seller).

Usage:
    python analyze_benchmark_results.py --csv logs/janus_full_benchmark_20260213_1721_price_domain.csv
    python analyze_benchmark_results.py --csv logs/janus_full_benchmark_*.csv --output_dir analysis_plots
"""

import argparse
import os
import sys
import glob
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init

init(autoreset=True)


@dataclass
class StrategySummary:
    """Summary statistics for a strategy."""
    strategy: str
    total_episodes: int
    agreements: int
    agreement_rate: float
    avg_janus_utility: float
    avg_janus_norm: float
    avg_opponent_utility: float
    avg_opponent_norm: float
    avg_turns: float
    janus_wins: int
    opponent_wins: int
    ties: int

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for plotting by creating long-format DataFrames.
    
    Returns:
        (janus_as_seller_df, janus_as_buyer_df)
    """
    # Filter for agreements only (where utilities are meaningful)
    df_agreements = df[df['agreement'] == True].copy()
    
    # Filter out micro strategies
    df_agreements = df_agreements[~df_agreements['strategy'].str.startswith('micro_')].copy()
    
    # Janus as Seller: Janus plays seller role
    janus_seller = df_agreements[df_agreements['janus_role'] == 'seller'].copy()
    janus_seller_data = []
    for _, row in janus_seller.iterrows():
        janus_seller_data.append({
            'strategy': row['strategy'],
            'agent': 'Janus (Seller)',
            'utility': row['janus_utility'],
            'norm_utility': row['janus_norm_utility']
        })
        janus_seller_data.append({
            'strategy': row['strategy'],
            'agent': 'Opponent (Buyer)',
            'utility': row['opponent_utility'],
            'norm_utility': row['opponent_norm_utility']
        })
    
    # Janus as Buyer: Janus plays buyer role
    janus_buyer = df_agreements[df_agreements['janus_role'] == 'buyer'].copy()
    janus_buyer_data = []
    for _, row in janus_buyer.iterrows():
        janus_buyer_data.append({
            'strategy': row['strategy'],
            'agent': 'Janus (Buyer)',
            'utility': row['janus_utility'],
            'norm_utility': row['janus_norm_utility']
        })
        janus_buyer_data.append({
            'strategy': row['strategy'],
            'agent': 'Opponent (Seller)',
            'utility': row['opponent_utility'],
            'norm_utility': row['opponent_norm_utility']
        })
    
    janus_seller_df = pd.DataFrame(janus_seller_data) if janus_seller_data else pd.DataFrame()
    janus_buyer_df = pd.DataFrame(janus_buyer_data) if janus_buyer_data else pd.DataFrame()
    
    return janus_seller_df, janus_buyer_df


def create_strategy_violin_plots(
    df_seller: pd.DataFrame,
    df_buyer: pd.DataFrame,
    output_dir: str,
    prefix: str = ""
):
    """
    Create violin plots for each strategy comparing Janus vs opponent utilities.
    
    Creates two plots per strategy:
    - Janus as Seller vs Opponent (Buyer)
    - Janus as Buyer vs Opponent (Seller)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique strategies from both dataframes
    strategies_seller = set(df_seller['strategy'].unique()) if not df_seller.empty else set()
    strategies_buyer = set(df_buyer['strategy'].unique()) if not df_buyer.empty else set()
    all_strategies = sorted(strategies_seller | strategies_buyer)
    
    print(f"\nGenerating plots for {len(all_strategies)} strategies...")
    
    for strategy in all_strategies:
        print(f"  Creating plots for: {strategy}")
        
        # Create figure with 1 or 2 subplots depending on data availability
        has_seller_data = strategy in strategies_seller and not df_seller[df_seller['strategy'] == strategy].empty
        has_buyer_data = strategy in strategies_buyer and not df_buyer[df_buyer['strategy'] == strategy].empty
        
        num_plots = (1 if has_seller_data else 0) + (1 if has_buyer_data else 0)
        if num_plots == 0:
            continue
        
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Janus as Seller
        if has_seller_data:
            data = df_seller[df_seller['strategy'] == strategy]
            ax = axes[plot_idx]
            
            # Create violin plot
            sns.violinplot(data=data, x='agent', y='norm_utility', ax=ax, palette=['#2ecc71', '#e74c3c'])
            ax.set_title(f'Janus as Seller vs {strategy}\n(Buyer opponent)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Agent', fontsize=10)
            ax.set_ylabel('Normalized Utility (% of ZOPA)', fontsize=10)
            ax.set_ylim(0, 1)
            
            # Add mean values as text
            for i, agent_type in enumerate(['Janus (Seller)', 'Opponent (Buyer)']):
                agent_data = data[data['agent'] == agent_type]['norm_utility']
                if not agent_data.empty:
                    mean_val = agent_data.mean()
                    ax.text(i, mean_val + 0.05, f'μ={mean_val:.2%}', 
                           ha='center', fontsize=9, fontweight='bold')
            
            plot_idx += 1
        
        # Plot 2: Janus as Buyer
        if has_buyer_data:
            data = df_buyer[df_buyer['strategy'] == strategy]
            ax = axes[plot_idx]
            
            # Create violin plot
            sns.violinplot(data=data, x='agent', y='norm_utility', ax=ax, palette=['#2ecc71', '#e74c3c'])
            ax.set_title(f'Janus as Buyer vs {strategy}\n(Seller opponent)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Agent', fontsize=10)
            ax.set_ylabel('Normalized Utility (% of ZOPA)', fontsize=10)
            ax.set_ylim(0, 1)
            
            # Add mean values as text
            for i, agent_type in enumerate(['Janus (Buyer)', 'Opponent (Seller)']):
                agent_data = data[data['agent'] == agent_type]['norm_utility']
                if not agent_data.empty:
                    mean_val = agent_data.mean()
                    ax.text(i, mean_val + 0.05, f'μ={mean_val:.2%}', 
                           ha='center', fontsize=9, fontweight='bold')
        
        plt.suptitle(f'Utility Distribution: Janus vs {strategy}', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        safe_strategy_name = strategy.replace(' ', '_').replace('/', '_')
        filename = f"{prefix}{safe_strategy_name}_violin.png" if prefix else f"{safe_strategy_name}_violin.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filepath}")
    
    print(f"\n{Fore.GREEN}All plots saved to: {output_dir}{Fore.RESET}")


def create_overall_summary_plot(
    summaries: List[StrategySummary],
    output_dir: str,
    prefix: str = ""
):
    """Create an overall summary plot comparing all strategies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = [s.strategy for s in summaries]
    x_pos = range(len(strategies))
    
    # Plot 1: Agreement Rate
    ax = axes[0, 0]
    agr_rates = [s.agreement_rate for s in summaries]
    bars = ax.bar(x_pos, agr_rates, color='#3498db')
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_title('Agreement Rate by Strategy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 100)
    
    # Plot 2: Janus Normalized Utility
    ax = axes[0, 1]
    janus_norms = [s.avg_janus_norm * 100 for s in summaries]
    bars = ax.bar(x_pos, janus_norms, color='#2ecc71')
    ax.set_ylabel('Normalized Utility (%)')
    ax.set_title('Janus Average Utility by Strategy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Fair split')
    ax.legend()
    
    # Plot 3: Win/Tie/Loss Count
    ax = axes[1, 0]
    janus_wins = [s.janus_wins for s in summaries]
    ties = [s.ties for s in summaries]
    opp_wins = [s.opponent_wins for s in summaries]
    
    width = 0.25
    ax.bar([x - width for x in x_pos], janus_wins, width, label='Janus Wins', color='#2ecc71')
    ax.bar(x_pos, ties, width, label='Ties', color='#f39c12')
    ax.bar([x + width for x in x_pos], opp_wins, width, label='Opponent Wins', color='#e74c3c')
    
    ax.set_ylabel('Count')
    ax.set_title('Win/Tie/Loss Distribution')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.legend()
    
    # Plot 4: Average Turns to Agreement
    ax = axes[1, 1]
    avg_turns = [s.avg_turns for s in summaries]
    bars = ax.bar(x_pos, avg_turns, color='#9b59b6')
    ax.set_ylabel('Average Turns')
    ax.set_title('Average Turns to Agreement')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    
    plt.suptitle('Janus Benchmark Summary - All Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{prefix}overall_summary.png" if prefix else "overall_summary.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{Fore.GREEN}Overall summary plot saved: {filepath}{Fore.RESET}")


def calculate_strategy_summary(strategy: str, results: List[Dict]) -> StrategySummary:
    """Calculate summary statistics for a strategy."""
    total = len(results)
    agreements = sum(1 for r in results if r['agreement'])
    
    if agreements == 0:
        return StrategySummary(
            strategy=strategy,
            total_episodes=total,
            agreements=0,
            agreement_rate=0.0,
            avg_janus_utility=0.0,
            avg_janus_norm=0.0,
            avg_opponent_utility=0.0,
            avg_opponent_norm=0.0,
            avg_turns=0.0,
            janus_wins=0,
            opponent_wins=0,
            ties=0
        )
    
    agreement_rate = agreements / total * 100
    
    janus_utils = [r['janus_norm_utility'] for r in results if r['agreement']]
    opponent_utils = [r['opponent_norm_utility'] for r in results if r['agreement']]
    turns_list = [r['turns'] for r in results]
    
    # Count wins (who gets more than 50% of zopa)
    janus_wins = sum(1 for r in results if r['agreement'] and r['janus_norm_utility'] > r['opponent_norm_utility'] + 0.01)
    opponent_wins = sum(1 for r in results if r['agreement'] and r['opponent_norm_utility'] > r['janus_norm_utility'] + 0.01)
    ties = agreements - janus_wins - opponent_wins
    
    return StrategySummary(
        strategy=strategy,
        total_episodes=total,
        agreements=agreements,
        agreement_rate=agreement_rate,
        avg_janus_utility=sum(r['janus_utility'] for r in results if r['agreement']) / agreements,
        avg_janus_norm=sum(janus_utils) / agreements,
        avg_opponent_utility=sum(r['opponent_utility'] for r in results if r['agreement']) / agreements,
        avg_opponent_norm=sum(opponent_utils) / agreements,
        avg_turns=sum(turns_list) / total,
        janus_wins=janus_wins,
        opponent_wins=opponent_wins,
        ties=ties
    )


def print_summary_table(summaries: List[StrategySummary]):
    """Print formatted summary table."""
    print(f"\n{Fore.CYAN}{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}{Fore.RESET}\n")
    
    print(f"{'Strategy':<25} {'Ep':>4} {'Agr':>4} {'Agr%':>6} {'Jan%':>7} {'Opp%':>7} {'Ties':>5} {'Turns':>6}")
    print("-" * 100)
    
    total_episodes = 0
    total_agreements = 0
    total_janus_wins = 0
    total_opponent_wins = 0
    total_ties = 0
    
    for s in summaries:
        print(f"{s.strategy:<25} {s.total_episodes:>4} {s.agreements:>4} {s.agreement_rate:>6.1f} "
              f"{s.avg_janus_norm*100:>6.1f}% {s.avg_opponent_norm*100:>6.1f}% {s.ties:>5} {s.avg_turns:>6.1f}")
        
        total_episodes += s.total_episodes
        total_agreements += s.agreements
        total_janus_wins += s.janus_wins
        total_opponent_wins += s.opponent_wins
        total_ties += s.ties
    
    print("-" * 100)
    overall_agr = total_agreements / total_episodes * 100 if total_episodes > 0 else 0
    
    # Calculate overall averages weighted by agreements
    janus_sum = sum(s.avg_janus_norm * s.agreements for s in summaries)
    opp_sum = sum(s.avg_opponent_norm * s.agreements for s in summaries)
    overall_janus = janus_sum / total_agreements if total_agreements > 0 else 0
    overall_opp = opp_sum / total_agreements if total_agreements > 0 else 0
    
    print(f"{'OVERALL':<25} {total_episodes:>4} {total_agreements:>4} {overall_agr:>6.1f} "
          f"{overall_janus*100:>6.1f}% {overall_opp*100:>6.1f}% {total_ties:>5}")
    
    print(f"\n{Fore.GREEN}Janus wins: {total_janus_wins} | Opponent wins: {total_opponent_wins} | Ties: {total_ties}{Fore.RESET}")


def main():
    """Main analysis entry point."""
    
    parser = argparse.ArgumentParser(
        description="Analyze Janus Benchmark Results"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to benchmark CSV file (or glob pattern)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="benchmark_analysis",
        help="Directory to save plots (default: benchmark_analysis)"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for output filenames"
    )
    args = parser.parse_args()
    
    # Handle glob patterns
    if '*' in args.csv:
        csv_files = glob.glob(args.csv)
        if not csv_files:
            print(f"No files found matching: {args.csv}")
            return
        csv_path = csv_files[0]  # Use first match
        print(f"Found {len(csv_files)} files, using: {csv_path}")
    else:
        csv_path = args.csv
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    # Load data
    print(f"\nLoading results from: {csv_path}")
    df = load_results(csv_path)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("\nPreparing data...")
    df_seller, df_buyer = prepare_data(df)
    
    # Calculate summaries
    print("Calculating summaries...")
    
    # Filter out micro strategies for summary
    df_filtered = df[~df['strategy'].str.startswith('micro_')].copy()
    
    # Group results by strategy for summary
    results_by_strategy: Dict[str, List[Dict]] = {}
    for _, row in df_filtered.iterrows():
        strategy = row['strategy']
        
        if strategy not in results_by_strategy:
            results_by_strategy[strategy] = []
        
        results_by_strategy[strategy].append({
            'episode_id': row['episode_id'],
            'strategy': strategy,
            'agreement': row['agreement'],
            'janus_utility': row['janus_utility'],
            'opponent_utility': row['opponent_utility'],
            'janus_norm_utility': row['janus_norm_utility'],
            'opponent_norm_utility': row['opponent_norm_utility'],
            'turns': row['turns'],
        })
    
    summaries = []
    for strategy, results in sorted(results_by_strategy.items()):
        summary = calculate_strategy_summary(strategy, results)
        summaries.append(summary)
    
    # Create overall summary plot
    print("\nCreating overall summary plot...")
    create_overall_summary_plot(summaries, args.output_dir, args.prefix)
    
    # Create violin plots for each strategy
    print("\nCreating strategy violin plots...")
    create_strategy_violin_plots(df_seller, df_buyer, args.output_dir, args.prefix)
    
    # Print summary table
    print_summary_table(summaries)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Plots saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
