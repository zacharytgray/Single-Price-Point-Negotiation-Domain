"""
Rank Strategies by Utility

This script analyzes a dataset (JSONL format) and ranks strategies by their
utility performance. It calculates both raw utility and normalized utility
(percentage of ZOPA captured) for each strategy.

Usage:
    python rank_strategies_by_utility.py --dataset datasets/price_domain_v8.jsonl
    python rank_strategies_by_utility.py --dataset datasets/price_domain_v8.jsonl --output strategy_rankings.json
    python rank_strategies_by_utility.py --dataset datasets/price_domain_v8.jsonl --min-episodes 10
    python rank_strategies_by_utility.py --dataset datasets/price_domain_v8.jsonl --combined
    python rank_strategies_by_utility.py --dataset datasets/price_domain_v8.jsonl --combined --combined-output combined_rankings.json
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class EpisodeResult:
    """Result of a single negotiation episode."""
    trajectory_id: str
    buyer_strategy: str
    seller_strategy: str
    buyer_max: float
    seller_min: float
    zopa_width: float
    agreement: bool
    final_price: Optional[float]
    num_turns: int
    rho: float


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""
    strategy: str
    role: str  # 'buyer' or 'seller'
    episodes: int = 0
    agreements: int = 0
    total_utility: float = 0.0
    total_norm_utility: float = 0.0
    utilities: List[float] = field(default_factory=list)
    norm_utilities: List[float] = field(default_factory=list)
    
    @property
    def avg_utility(self) -> float:
        return self.total_utility / self.episodes if self.episodes > 0 else 0.0
    
    @property
    def avg_norm_utility(self) -> float:
        return self.total_norm_utility / self.episodes if self.episodes > 0 else 0.0
    
    @property
    def agreement_rate(self) -> float:
        return self.agreements / self.episodes if self.episodes > 0 else 0.0
    
    @property
    def std_utility(self) -> float:
        if len(self.utilities) < 2:
            return 0.0
        mean = self.avg_utility
        variance = sum((x - mean) ** 2 for x in self.utilities) / len(self.utilities)
        return variance ** 0.5
    
    @property
    def std_norm_utility(self) -> float:
        if len(self.norm_utilities) < 2:
            return 0.0
        mean = self.avg_norm_utility
        variance = sum((x - mean) ** 2 for x in self.norm_utilities) / len(self.norm_utilities)
        return variance ** 0.5


@dataclass
class CombinedStrategyStats:
    """Statistics for a strategy combining both buyer and seller roles."""
    strategy: str
    total_episodes: int = 0
    total_agreements: int = 0
    total_utility: float = 0.0
    total_norm_utility: float = 0.0
    buyer_episodes: int = 0
    seller_episodes: int = 0
    buyer_avg_norm_utility: float = 0.0
    seller_avg_norm_utility: float = 0.0
    
    @property
    def agreement_rate(self) -> float:
        return self.total_agreements / self.total_episodes if self.total_episodes > 0 else 0.0
    
    @property
    def avg_utility(self) -> float:
        return self.total_utility / self.total_episodes if self.total_episodes > 0 else 0.0
    
    @property
    def avg_norm_utility(self) -> float:
        return self.total_norm_utility / self.total_episodes if self.total_episodes > 0 else 0.0
    
    @property
    def role_balance(self) -> float:
        """Returns ratio of buyer episodes to total (0.5 = balanced, 1.0 = all buyer, 0.0 = all seller)."""
        return self.buyer_episodes / self.total_episodes if self.total_episodes > 0 else 0.0


def combine_strategy_stats(strategy_stats: Dict[str, StrategyStats]) -> Dict[str, CombinedStrategyStats]:
    """Combine buyer and seller stats for each strategy into unified stats."""
    combined: Dict[str, CombinedStrategyStats] = {}
    
    for key, stats in strategy_stats.items():
        strategy_name = stats.strategy
        
        if strategy_name not in combined:
            combined[strategy_name] = CombinedStrategyStats(strategy=strategy_name)
        
        c = combined[strategy_name]
        c.total_episodes += stats.episodes
        c.total_agreements += stats.agreements
        c.total_utility += stats.total_utility
        c.total_norm_utility += stats.total_norm_utility
        
        if stats.role == 'buyer':
            c.buyer_episodes = stats.episodes
            c.buyer_avg_norm_utility = stats.avg_norm_utility
        else:
            c.seller_episodes = stats.episodes
            c.seller_avg_norm_utility = stats.avg_norm_utility
    
    return combined


def load_dataset(dataset_path: str) -> List[dict]:
    """Load the dataset from JSONL file."""
    data = []
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Loading dataset from {dataset_path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} records")
    return data


def extract_episodes(data: List[dict]) -> List[EpisodeResult]:
    """Extract unique episodes from the dataset records."""
    episodes = {}
    
    for record in data:
        trajectory_id = record.get('trajectory_id')
        if not trajectory_id:
            continue
        
        # Only process each episode once (use first turn to get all info)
        if trajectory_id in episodes:
            continue
        
        # Get turn 1 info to identify buyer's strategy
        turn = record.get('turn', 0)
        if turn != 1:
            continue
        
        # Get episode outcome from meta
        meta = record.get('meta', {})
        episode_outcome = meta.get('episode_outcome', {})
        
        if not episode_outcome:
            continue
        
        # Get buyer and seller strategies from episode records
        # Need to find both buyer and seller strategies from the data
        buyer_strategy = None
        seller_strategy = None
        
        # For now, extract from current record
        agent_info = record.get('agent', {})
        role = record.get('state', {}).get('role', '')
        strategy = agent_info.get('strategy', 'unknown')
        
        if role == 'buyer':
            buyer_strategy = strategy
        elif role == 'seller':
            seller_strategy = strategy
        
        # Create episode with available info
        # Note: we'll need to find both strategies
        episodes[trajectory_id] = {
            'trajectory_id': trajectory_id,
            'buyer_strategy': buyer_strategy,
            'seller_strategy': seller_strategy,
            'buyer_max': meta.get('buyer_max', 0),
            'seller_min': meta.get('seller_min', 0),
            'zopa_width': meta.get('zopa_width', 0),
            'agreement': episode_outcome.get('agreement_bool', False),
            'final_price': episode_outcome.get('price'),
            'num_turns': episode_outcome.get('num_turns', 0),
            'rho': record.get('rho', 0),
        }
    
    # Second pass: find the other agent's strategy for each episode
    for record in data:
        trajectory_id = record.get('trajectory_id')
        if trajectory_id not in episodes:
            continue
        
        turn = record.get('turn', 0)
        if turn != 2:  # Second turn is the other agent
            continue
        
        agent_info = record.get('agent', {})
        role = record.get('state', {}).get('role', '')
        strategy = agent_info.get('strategy', 'unknown')
        
        if role == 'buyer' and not episodes[trajectory_id]['buyer_strategy']:
            episodes[trajectory_id]['buyer_strategy'] = strategy
        elif role == 'seller' and not episodes[trajectory_id]['seller_strategy']:
            episodes[trajectory_id]['seller_strategy'] = strategy
    
    # Convert to EpisodeResult objects
    results = []
    for ep in episodes.values():
        if not ep['buyer_strategy'] or not ep['seller_strategy']:
            continue
            
        results.append(EpisodeResult(
            trajectory_id=ep['trajectory_id'],
            buyer_strategy=ep['buyer_strategy'],
            seller_strategy=ep['seller_strategy'],
            buyer_max=ep['buyer_max'],
            seller_min=ep['seller_min'],
            zopa_width=ep['zopa_width'],
            agreement=ep['agreement'],
            final_price=ep['final_price'],
            num_turns=ep['num_turns'],
            rho=ep['rho']
        ))
    
    return results


def calculate_utility(episode: EpisodeResult, role: str) -> Tuple[float, float]:
    """
    Calculate utility for a given role in an episode.
    
    Returns:
        Tuple of (raw_utility, normalized_utility)
    """
    if not episode.agreement or episode.final_price is None:
        return 0.0, 0.0
    
    if role == 'buyer':
        raw_utility = episode.buyer_max - episode.final_price
    else:  # seller
        raw_utility = episode.final_price - episode.seller_min
    
    # Normalize by ZOPA width
    if episode.zopa_width > 0:
        norm_utility = raw_utility / episode.zopa_width
    else:
        norm_utility = 0.0
    
    return raw_utility, norm_utility


def analyze_strategies(episodes: List[EpisodeResult], min_episodes: int = 1) -> Dict[str, StrategyStats]:
    """Analyze all strategies and calculate statistics."""
    strategy_stats: Dict[str, StrategyStats] = {}
    
    def get_stats(strategy: str, role: str) -> StrategyStats:
        key = f"{strategy}_{role}"
        if key not in strategy_stats:
            strategy_stats[key] = StrategyStats(strategy=strategy, role=role)
        return strategy_stats[key]
    
    for episode in episodes:
        # Process buyer
        buyer_stats = get_stats(episode.buyer_strategy, 'buyer')
        buyer_stats.episodes += 1
        if episode.agreement:
            buyer_stats.agreements += 1
            raw_util, norm_util = calculate_utility(episode, 'buyer')
            buyer_stats.total_utility += raw_util
            buyer_stats.total_norm_utility += norm_util
            buyer_stats.utilities.append(raw_util)
            buyer_stats.norm_utilities.append(norm_util)
        
        # Process seller
        seller_stats = get_stats(episode.seller_strategy, 'seller')
        seller_stats.episodes += 1
        if episode.agreement:
            seller_stats.agreements += 1
            raw_util, norm_util = calculate_utility(episode, 'seller')
            seller_stats.total_utility += raw_util
            seller_stats.total_norm_utility += norm_util
            seller_stats.utilities.append(raw_util)
            seller_stats.norm_utilities.append(norm_util)
    
    # Filter by minimum episodes
    filtered_stats = {
        key: stats for key, stats in strategy_stats.items()
        if stats.episodes >= min_episodes
    }
    
    return filtered_stats


def print_rankings(strategy_stats: Dict[str, StrategyStats], sort_by: str = 'norm_utility'):
    """Print strategy rankings to console."""
    
    # Sort strategies
    if sort_by == 'norm_utility':
        sorted_stats = sorted(strategy_stats.values(), key=lambda x: x.avg_norm_utility, reverse=True)
        metric_name = "Avg Normalized Utility"
    elif sort_by == 'utility':
        sorted_stats = sorted(strategy_stats.values(), key=lambda x: x.avg_utility, reverse=True)
        metric_name = "Avg Raw Utility"
    elif sort_by == 'agreement_rate':
        sorted_stats = sorted(strategy_stats.values(), key=lambda x: x.agreement_rate, reverse=True)
        metric_name = "Agreement Rate"
    else:
        sorted_stats = sorted(strategy_stats.values(), key=lambda x: x.avg_norm_utility, reverse=True)
        metric_name = "Avg Normalized Utility"
    
    # Print header
    print("\n" + "=" * 100)
    print(f"STRATEGY RANKINGS (sorted by {metric_name})")
    print("=" * 100)
    print(f"{'Rank':<6} {'Strategy':<30} {'Role':<8} {'Episodes':<10} {'Agree%':<8} {'Avg Util':<12} {'Avg Norm%':<12} {'Std Norm%':<12}")
    print("-" * 100)
    
    # Print each strategy
    for rank, stats in enumerate(sorted_stats, 1):
        print(f"{rank:<6} {stats.strategy:<30} {stats.role:<8} {stats.episodes:<10} "
              f"{stats.agreement_rate*100:>6.1f}%  {stats.avg_utility:>9.2f}    "
              f"{stats.avg_norm_utility*100:>8.2f}%    {stats.std_norm_utility*100:>8.2f}%")
    
    print("=" * 100)
    print(f"\nTotal unique strategies analyzed: {len(sorted_stats)}")
    
    # Print role-specific rankings
    print("\n" + "=" * 100)
    print("BUYER STRATEGIES RANKING (by Normalized Utility)")
    print("=" * 100)
    buyer_stats = [s for s in sorted_stats if s.role == 'buyer']
    for rank, stats in enumerate(buyer_stats, 1):
        print(f"{rank:<6} {stats.strategy:<30} {stats.episodes:<10} "
              f"{stats.agreement_rate*100:>6.1f}%  {stats.avg_norm_utility*100:>8.2f}%")
    
    print("\n" + "=" * 100)
    print("SELLER STRATEGIES RANKING (by Normalized Utility)")
    print("=" * 100)
    seller_stats = [s for s in sorted_stats if s.role == 'seller']
    for rank, stats in enumerate(seller_stats, 1):
        print(f"{rank:<6} {stats.strategy:<30} {stats.episodes:<10} "
              f"{stats.agreement_rate*100:>6.1f}%  {stats.avg_norm_utility*100:>8.2f}%")


def print_combined_rankings(combined_stats: Dict[str, CombinedStrategyStats], sort_by: str = 'norm_utility'):
    """Print combined strategy rankings (both roles averaged) to console."""

    # Sort strategies
    if sort_by == 'norm_utility':
        sorted_stats = sorted(combined_stats.values(), key=lambda x: x.avg_norm_utility, reverse=True)
        metric_name = "Avg Normalized Utility"
    elif sort_by == 'utility':
        sorted_stats = sorted(combined_stats.values(), key=lambda x: x.avg_utility, reverse=True)
        metric_name = "Avg Raw Utility"
    elif sort_by == 'agreement_rate':
        sorted_stats = sorted(combined_stats.values(), key=lambda x: x.agreement_rate, reverse=True)
        metric_name = "Agreement Rate"
    else:
        sorted_stats = sorted(combined_stats.values(), key=lambda x: x.avg_norm_utility, reverse=True)
        metric_name = "Avg Normalized Utility"

    # Print header
    print("\n" + "=" * 100)
    print(f"COMBINED STRATEGY RANKINGS - Buyer + Seller (sorted by {metric_name})")
    print("=" * 100)
    print(f"{'Rank':<6} {'Strategy':<30} {'Episodes':<10} {'Agree%':<8} {'Avg Norm%':<12} {'Buy%':<8} {'Sell%':<8} {'Balance':<8}")
    print("-" * 100)

    # Print each strategy
    for rank, stats in enumerate(sorted_stats, 1):
        # Calculate buyer/seller percentages
        buy_pct = stats.buyer_avg_norm_utility * 100 if stats.buyer_episodes > 0 else 0
        sell_pct = stats.seller_avg_norm_utility * 100 if stats.seller_episodes > 0 else 0
        balance = stats.role_balance
        balance_str = f"{balance:.2f}" if 0 < stats.buyer_episodes and 0 < stats.seller_episodes else "N/A"

        print(f"{rank:<6} {stats.strategy:<30} {stats.total_episodes:<10} "
              f"{stats.agreement_rate*100:>6.1f}%  {stats.avg_norm_utility*100:>8.2f}%  "
              f"{buy_pct:>6.1f}%  {sell_pct:>6.1f}%  {balance_str:>8}")

    print("=" * 100)
    print(f"\nTotal unique strategies analyzed: {len(sorted_stats)}")
    print("\nBalance: 0.50 = equal buyer/seller episodes, 1.00 = all buyer, 0.00 = all seller")


def export_combined_results(combined_stats: Dict[str, CombinedStrategyStats], output_path: str):
    """Export combined rankings to JSON file."""

    # Convert to serializable format
    results = {
        "strategies": []
    }

    for stats in combined_stats.values():
        results["strategies"].append({
            "strategy": stats.strategy,
            "total_episodes": stats.total_episodes,
            "total_agreements": stats.total_agreements,
            "agreement_rate": stats.agreement_rate,
            "avg_utility": stats.avg_utility,
            "avg_norm_utility": stats.avg_norm_utility,
            "buyer_episodes": stats.buyer_episodes,
            "seller_episodes": stats.seller_episodes,
            "buyer_avg_norm_utility": stats.buyer_avg_norm_utility,
            "seller_avg_norm_utility": stats.seller_avg_norm_utility,
            "role_balance": stats.role_balance
        })

    # Sort by normalized utility
    results["strategies"].sort(key=lambda x: x["avg_norm_utility"], reverse=True)

    # Add metadata
    results["metadata"] = {
        "total_strategies": len(results["strategies"]),
        "sorted_by": "avg_norm_utility",
        "combined_roles": True
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nCombined results exported to: {output_path}")


def export_results(strategy_stats: Dict[str, StrategyStats], output_path: str):
    """Export rankings to JSON file."""
    
    # Convert to serializable format
    results = {
        "strategies": []
    }
    
    for key, stats in strategy_stats.items():
        results["strategies"].append({
            "strategy": stats.strategy,
            "role": stats.role,
            "episodes": stats.episodes,
            "agreements": stats.agreements,
            "agreement_rate": stats.agreement_rate,
            "avg_utility": stats.avg_utility,
            "avg_norm_utility": stats.avg_norm_utility,
            "std_utility": stats.std_utility,
            "std_norm_utility": stats.std_norm_utility,
            "total_utility": stats.total_utility,
            "total_norm_utility": stats.total_norm_utility
        })
    
    # Sort by normalized utility
    results["strategies"].sort(key=lambda x: x["avg_norm_utility"], reverse=True)
    
    # Add metadata
    results["metadata"] = {
        "total_strategies": len(results["strategies"]),
        "sorted_by": "avg_norm_utility"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank negotiation strategies by utility performance"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for detailed results"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=['norm_utility', 'utility', 'agreement_rate'],
        default='norm_utility',
        help="Metric to sort strategies by (default: norm_utility)"
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=1,
        help="Minimum number of episodes required for a strategy to be included (default: 1)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Show combined rankings (merge buyer and seller into single strategy ranking)"
    )
    parser.add_argument(
        "--combined-output",
        type=str,
        help="Output JSON file path for combined results (both roles merged)"
    )

    args = parser.parse_args()

    # Load and process data
    data = load_dataset(args.dataset)
    episodes = extract_episodes(data)

    print(f"\nExtracted {len(episodes)} unique episodes")

    # Analyze strategies
    strategy_stats = analyze_strategies(episodes, min_episodes=args.min_episodes)

    print(f"Found {len(strategy_stats)} strategy-role combinations (min_episodes={args.min_episodes})")

    # Print rankings (role-separated)
    print_rankings(strategy_stats, sort_by=args.sort_by)

    # Export role-separated results if requested
    if args.output:
        export_results(strategy_stats, args.output)

    # Combined rankings (merge buyer and seller)
    if args.combined or args.combined_output:
        combined_stats = combine_strategy_stats(strategy_stats)
        print_combined_rankings(combined_stats, sort_by=args.sort_by)

        if args.combined_output:
            export_combined_results(combined_stats, args.combined_output)


if __name__ == "__main__":
    main()
