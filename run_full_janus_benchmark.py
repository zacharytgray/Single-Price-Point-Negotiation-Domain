"""
Full Janus Benchmark - Comprehensive evaluation against deterministic strategies.

Tests Janus agent against all unique deterministic strategies (50 episodes each),
excluding random strategies and margin variants (duplicates).

Usage:
    python run_full_janus_benchmark.py
    python run_full_janus_benchmark.py --janus_rho 0.5 --episodes_per_strategy 50
    python run_full_janus_benchmark.py --buyer_rho 0.2 --seller_rho 0.8
"""

import asyncio
import sys
import os
import random
import time
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import csv
import json
import argparse
from colorama import Fore, init

init(autoreset=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    MAX_TURNS,
    DEFAULT_BUYER_MAX_MEAN,
    DEFAULT_BUYER_MAX_STD,
    FIXED_ZOPA_WIDTH,
    PRICE_RANGE_LOW,
    PRICE_RANGE_HIGH,
    JANUS_ADAPTER_PATH,
    JANUS_MODEL_PATH
)
from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.agents.janus_agent import JanusAgent
from src.core.price_structures import PriceState
from src.domain.single_issue_price_domain import SingleIssuePriceDomain


# Benchmark Configuration
DEFAULT_EPISODES_PER_STRATEGY = 50
LOG_DIR = "logs"

# Strategies to exclude (random, margin variants, and micro strategies)
EXCLUDED_STRATEGIES = {
    'random_zopa',  # Random strategy
    # Margin variants (same logic with different param)
    'boulware_very_conceding_margin',
    'boulware_conceding_margin',
    'boulware_firm_margin',
    'boulware_hard_margin',
    'hardliner_margin',
    # Micro strategies
    'micro_fine',
    'micro_moderate',
    'micro_coarse',
}


def get_benchmark_strategies() -> List[str]:
    """Get list of unique deterministic strategies for benchmarking."""
    all_strategies = set(STRATEGY_REGISTRY.keys())
    return sorted(list(all_strategies - EXCLUDED_STRATEGIES))


@dataclass
class BenchmarkResult:
    """Result for a single episode."""
    episode_id: int
    strategy: str
    janus_role: str
    agreement: bool
    final_price: Optional[float]
    turns: int
    janus_utility: float
    janus_norm_utility: float
    opponent_utility: float
    opponent_norm_utility: float
    buyer_max: float
    seller_min: float
    zopa_width: float
    starting_agent: str
    timestamp: str
    offer_history: str  # JSON string of offer history


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
    janus_wins: int  # Janus gets >50% of zopa
    opponent_wins: int  # Opponent gets >50% of zopa
    ties: int  # Both get ~50%


class BenchmarkLogger:
    """Logger for benchmark results."""
    
    def __init__(self, session_name: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = f"{session_name}_{timestamp}"
        self.csv_filename = f"{session_name}_{timestamp}.csv"
        self.csv_filepath = os.path.join(LOG_DIR, self.csv_filename)
        self.json_filepath = os.path.join(LOG_DIR, f"{session_name}_{timestamp}_summary.json")
        
        os.makedirs(LOG_DIR, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def log_episode(self, result: BenchmarkResult):
        """Log a single episode result."""
        self.results.append(result)
        self._append_to_csv(result)
    
    def _append_to_csv(self, result: BenchmarkResult):
        """Append result to CSV file."""
        file_exists = os.path.exists(self.csv_filepath)
        
        with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header
                writer.writerow([
                    'episode_id', 'strategy', 'janus_role', 'agreement',
                    'final_price', 'turns', 'janus_utility', 'janus_norm_utility',
                    'opponent_utility', 'opponent_norm_utility',
                    'buyer_max', 'seller_min', 'zopa_width', 'starting_agent', 'timestamp',
                    'offer_history'
                ])
            writer.writerow([
                result.episode_id, result.strategy, result.janus_role, result.agreement,
                result.final_price, result.turns, result.janus_utility, result.janus_norm_utility,
                result.opponent_utility, result.opponent_norm_utility,
                result.buyer_max, result.seller_min, result.zopa_width, result.starting_agent,
                result.timestamp, result.offer_history
            ])
    
    def save_summary(self, summaries: List[StrategySummary]):
        """Save summary statistics to JSON."""
        summary_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(self.results),
            'strategies_tested': len(summaries),
            'janus_adapter_path': JANUS_ADAPTER_PATH,
            'janus_model_path': JANUS_MODEL_PATH,
            'strategy_summaries': [
                {
                    'strategy': s.strategy,
                    'total_episodes': s.total_episodes,
                    'agreements': s.agreements,
                    'agreement_rate': s.agreement_rate,
                    'avg_janus_utility': s.avg_janus_utility,
                    'avg_janus_norm': s.avg_janus_norm,
                    'avg_opponent_utility': s.avg_opponent_utility,
                    'avg_opponent_norm': s.avg_opponent_norm,
                    'avg_turns': s.avg_turns,
                    'janus_wins': s.janus_wins,
                    'opponent_wins': s.opponent_wins,
                    'ties': s.ties,
                }
                for s in summaries
            ]
        }
        
        with open(self.json_filepath, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def get_csv_path(self) -> str:
        return self.csv_filepath
    
    def get_json_path(self) -> str:
        return self.json_filepath


def create_price_state(
    timestep: int,
    max_turns: int,
    role: str,
    last_offer: Optional[float],
    history: List[Tuple[str, float]],
    reservation: float,
    buyer_max: float,
    seller_min: float
) -> PriceState:
    """Create a PriceState object for deterministic agent."""
    return PriceState(
        timestep=timestep,
        max_turns=max_turns,
        role=role,
        last_offer_price=last_offer,
        offer_history=list(history),
        effective_reservation_price=reservation,
        true_reservation_price=reservation,
        public_price_range=(PRICE_RANGE_LOW, PRICE_RANGE_HIGH)
    )


async def run_episode(
    janus_agent: JanusAgent,
    opponent: DeterministicPriceAgent,
    janus_role: str,
    buyer_max: float,
    seller_min: float,
    starting_agent: str,
    episode_id: int,
    strategy_name: str,
    logger: BenchmarkLogger,
    verbose: bool = False
) -> BenchmarkResult:
    """Run a single negotiation episode."""
    
    domain = SingleIssuePriceDomain()
    domain.reset(episode_id)
    domain.buyer_max = buyer_max
    domain.seller_min = seller_min
    
    # Update Janus price bounds
    janus_agent.p_low = PRICE_RANGE_LOW
    janus_agent.p_high = PRICE_RANGE_HIGH
    
    history: List[Tuple[str, float]] = []
    offer_history_log: List[Dict[str, Any]] = []  # Detailed offer history
    last_offer: Optional[float] = None
    
    opponent_role = "seller" if janus_role == "buyer" else "buyer"
    opponent_reservation = seller_min if opponent_role == "seller" else buyer_max
    
    janus_agent_num = 1 if janus_role == "buyer" else 2
    opponent_agent_num = 2 if opponent_role == "seller" else 1
    
    current_agent = starting_agent
    agreement_reached = False
    final_price = None
    turn = 0
    
    if verbose:
        print(f"\n  Episode {episode_id} | Janus={janus_role} vs {strategy_name} | Start={starting_agent}")
        print(f"  ZOPA: Seller Min=${seller_min:.2f}, Buyer Max=${buyer_max:.2f}")
    
    while turn < MAX_TURNS and not agreement_reached:
        turn += 1
        
        if current_agent == "janus":
            # Janus's turn
            janus_agent.domain_private_context = {
                "role": janus_role,
                "turn": turn,
                "max_turns": MAX_TURNS,
                "history": history,
                "max_willingness_to_pay": buyer_max if janus_role == "buyer" else None,
                "min_acceptable_price": seller_min if janus_role == "seller" else None,
            }
            
            response = await janus_agent.generate_response()
            action = domain.parse_agent_action(janus_agent_num, response)
            
            if action.action_type == "OFFER" and action.offer_content is not None:
                last_offer = action.offer_content
                history.append((janus_role, last_offer))
                offer_history_log.append({
                    'turn': turn,
                    'agent': 'janus',
                    'role': janus_role,
                    'action': 'OFFER',
                    'price': last_offer
                })
                domain.current_offer = last_offer
                domain.last_offerer_id = janus_agent_num
                
            elif action.action_type == "ACCEPT" and last_offer is not None:
                agreement_reached = True
                final_price = last_offer
                offer_history_log.append({
                    'turn': turn,
                    'agent': 'janus',
                    'role': janus_role,
                    'action': 'ACCEPT',
                    'price': last_offer
                })
                
            current_agent = "opponent"
            
        else:
            # Opponent's turn
            state = create_price_state(
                timestep=turn,
                max_turns=MAX_TURNS,
                role=opponent_role,
                last_offer=last_offer,
                history=history,
                reservation=opponent_reservation,
                buyer_max=buyer_max,
                seller_min=seller_min
            )
            
            action = opponent.propose_action(state)
            
            if action.type == "OFFER":
                last_offer = action.price
                history.append((opponent_role, last_offer))
                offer_history_log.append({
                    'turn': turn,
                    'agent': 'opponent',
                    'role': opponent_role,
                    'action': 'OFFER',
                    'price': last_offer
                })
                domain.current_offer = last_offer
                domain.last_offerer_id = opponent_agent_num
                
            elif action.type == "ACCEPT" and last_offer is not None:
                agreement_reached = True
                final_price = last_offer
                offer_history_log.append({
                    'turn': turn,
                    'agent': 'opponent',
                    'role': opponent_role,
                    'action': 'ACCEPT',
                    'price': last_offer
                })
            
            current_agent = "janus"
    
    # Calculate utilities
    zopa_width = buyer_max - seller_min
    
    if agreement_reached and final_price is not None:
        buyer_utility = buyer_max - final_price
        seller_utility = final_price - seller_min
        
        janus_utility = buyer_utility if janus_role == "buyer" else seller_utility
        opponent_utility = seller_utility if janus_role == "buyer" else buyer_utility
        
        janus_norm = janus_utility / zopa_width if zopa_width > 0 else 0
        opponent_norm = opponent_utility / zopa_width if zopa_width > 0 else 0
        
        if verbose:
            print(f"  AGREEMENT at ${final_price:.2f} | Janus={janus_norm:.2%} | Opp={opponent_norm:.2%}")
    else:
        janus_utility = 0.0
        opponent_utility = 0.0
        janus_norm = 0.0
        opponent_norm = 0.0
        final_price = None
        
        if verbose:
            print(f"  IMPASSE after {turn} turns")
    
    # Convert offer history to JSON string
    offer_history_json = json.dumps(offer_history_log)
    
    result = BenchmarkResult(
        episode_id=episode_id,
        strategy=strategy_name,
        janus_role=janus_role,
        agreement=agreement_reached,
        final_price=final_price,
        turns=turn,
        janus_utility=janus_utility,
        janus_norm_utility=janus_norm,
        opponent_utility=opponent_utility,
        opponent_norm_utility=opponent_norm,
        buyer_max=buyer_max,
        seller_min=seller_min,
        zopa_width=zopa_width,
        starting_agent=starting_agent,
        timestamp=datetime.now().isoformat(),
        offer_history=offer_history_json
    )
    
    logger.log_episode(result)
    return result


def calculate_strategy_summary(strategy: str, results: List[BenchmarkResult]) -> StrategySummary:
    """Calculate summary statistics for a strategy."""
    total = len(results)
    agreements = sum(1 for r in results if r.agreement)
    
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
    
    janus_utils = [r.janus_norm_utility for r in results if r.agreement]
    opponent_utils = [r.opponent_norm_utility for r in results if r.agreement]
    turns_list = [r.turns for r in results]
    
    # Count wins (who gets more than 50% of zopa)
    janus_wins = sum(1 for r in results if r.agreement and r.janus_norm_utility > r.opponent_norm_utility + 0.01)
    opponent_wins = sum(1 for r in results if r.agreement and r.opponent_norm_utility > r.janus_norm_utility + 0.01)
    ties = agreements - janus_wins - opponent_wins
    
    return StrategySummary(
        strategy=strategy,
        total_episodes=total,
        agreements=agreements,
        agreement_rate=agreement_rate,
        avg_janus_utility=sum(r.janus_utility for r in results if r.agreement) / agreements,
        avg_janus_norm=sum(janus_utils) / agreements,
        avg_opponent_utility=sum(r.opponent_utility for r in results if r.agreement) / agreements,
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


async def benchmark_strategy(
    strategy_name: str,
    episodes: int,
    janus_buyer: JanusAgent,
    janus_seller: JanusAgent,
    logger: BenchmarkLogger,
    verbose: bool = False
) -> List[BenchmarkResult]:
    """Run benchmark for a single strategy."""
    
    results = []
    
    for ep in range(1, episodes + 1):
        # Alternate Janus role
        janus_role = "buyer" if ep % 2 == 1 else "seller"
        janus_agent = janus_buyer if janus_role == "buyer" else janus_seller
        
        # Create opponent
        opponent_role = "seller" if janus_role == "buyer" else "buyer"
        opponent = DeterministicPriceAgent(
            agent_id=2 if opponent_role == "seller" else 1,
            strategy_name=strategy_name
        )
        
        # Randomize starting agent
        starting_agent = random.choice(["janus", "opponent"])
        
        # Generate ZOPA
        buyer_max = round(random.gauss(DEFAULT_BUYER_MAX_MEAN, DEFAULT_BUYER_MAX_STD), 2)
        seller_min = round(buyer_max - FIXED_ZOPA_WIDTH, 2)
        
        result = await run_episode(
            janus_agent=janus_agent,
            opponent=opponent,
            janus_role=janus_role,
            buyer_max=buyer_max,
            seller_min=seller_min,
            starting_agent=starting_agent,
            episode_id=ep,
            strategy_name=strategy_name,
            logger=logger,
            verbose=verbose
        )
        
        results.append(result)
        
        # Print progress every 10 episodes
        if ep % 10 == 0:
            print(f"    Progress: {ep}/{episodes} episodes completed")
    
    return results


async def main():
    """Main benchmark entry point."""
    
    parser = argparse.ArgumentParser(
        description="Full Janus Benchmark against Deterministic Strategies"
    )
    parser.add_argument(
        "--episodes_per_strategy", type=int, default=DEFAULT_EPISODES_PER_STRATEGY,
        help="Number of episodes per strategy (default: 50)"
    )
    parser.add_argument(
        "--buyer_rho", type=float, default=0.2,
        help="Rho value for Janus when playing as buyer (default: 0.2)"
    )
    parser.add_argument(
        "--seller_rho", type=float, default=0.8,
        help="Rho value for Janus when playing as seller (default: 0.8)"
    )
    parser.add_argument(
        "--janus_rho", type=float, default=None,
        help="Rho value for Janus in both roles (overrides buyer/seller rho)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed episode logs"
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+", default=None,
        help="Specific strategies to test (default: all non-excluded)"
    )
    args = parser.parse_args()
    
    # Handle janus_rho override
    if args.janus_rho is not None:
        buyer_rho = args.janus_rho
        seller_rho = args.janus_rho
    else:
        buyer_rho = args.buyer_rho
        seller_rho = args.seller_rho
    
    # Get strategies to test
    strategies = args.strategies if args.strategies else get_benchmark_strategies()
    
    print(f"{Fore.MAGENTA}{'='*80}")
    print("FULL JANUS BENCHMARK")
    print(f"{'='*80}{Fore.RESET}")
    print(f"\nConfiguration:")
    print(f"  Episodes per strategy: {args.episodes_per_strategy}")
    print(f"  Janus buyer rho: {buyer_rho}")
    print(f"  Janus seller rho: {seller_rho}")
    print(f"  Total strategies: {len(strategies)}")
    print(f"  Total episodes: {len(strategies) * args.episodes_per_strategy}")
    print(f"\nStrategies to test:")
    for i, s in enumerate(strategies, 1):
        print(f"  {i:2}. {s}")
    print()
    
    # Initialize logger
    logger = BenchmarkLogger("janus_full_benchmark")
    print(f"{Fore.GREEN}Logging to: {logger.get_csv_path()}{Fore.RESET}\n")
    
    # Create Janus agents (reused across all strategies)
    print(f"{Fore.YELLOW}Initializing Janus agents...{Fore.RESET}")
    
    janus_buyer = JanusAgent(
        agent_id=1,
        role="buyer",
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=buyer_rho
    )
    
    janus_seller = JanusAgent(
        agent_id=2,
        role="seller",
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=seller_rho
    )
    
    print(f"{Fore.GREEN}Janus agents ready!{Fore.RESET}\n")
    
    # Run benchmarks
    all_results: List[BenchmarkResult] = []
    summaries: List[StrategySummary] = []
    
    start_time = time.time()
    
    for i, strategy_name in enumerate(strategies, 1):
        print(f"\n{Fore.CYAN}[{i}/{len(strategies)}] Benchmarking against: {strategy_name}{Fore.RESET}")
        
        try:
            results = await benchmark_strategy(
                strategy_name=strategy_name,
                episodes=args.episodes_per_strategy,
                janus_buyer=janus_buyer,
                janus_seller=janus_seller,
                logger=logger,
                verbose=args.verbose
            )
            
            all_results.extend(results)
            summary = calculate_strategy_summary(strategy_name, results)
            summaries.append(summary)
            
            # Print mini summary
            print(f"  {Fore.GREEN}Complete:{Fore.RESET} Agr={summary.agreement_rate:.1f}% "
                  f"Janus={summary.avg_janus_norm:.2%} Opp={summary.avg_opponent_norm:.2%}")
            
        except Exception as e:
            print(f"{Fore.RED}  ERROR: {e}{Fore.RESET}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Print final summary
    print_summary_table(summaries)
    
    # Save summary
    logger.save_summary(summaries)
    
    # Final output
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}{Fore.RESET}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Episodes completed: {len(all_results)}/{len(strategies) * args.episodes_per_strategy}")
    print(f"\nResults saved:")
    print(f"  CSV: {logger.get_csv_path()}")
    print(f"  JSON: {logger.get_json_path()}")


if __name__ == "__main__":
    asyncio.run(main())
