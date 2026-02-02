"""
Run Janus HyperLoRA agent against deterministic strategy opponents.

This script pits the trained Janus agent against deterministic 
negotiation schedules:
- Boulware variants (very_conceding, conceding, linear, firm, hard)
- Price fixed (strict, loose)
- Tit-for-tat
- Linear standard
- Split difference
- Time dependent
- Hardliner
- MiCRO variants

Usage:
    python run_janus_vs_deterministic.py
    python run_janus_vs_deterministic.py --num_episodes 20 --janus_buyer_rho 0.3
"""

import asyncio
import sys
import os
import random
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import csv
import json
from colorama import Fore, init

init(autoreset=True)

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
from src.core.price_structures import PriceState, PriceAction
from src.domain.single_issue_price_domain import SingleIssuePriceDomain


# Configuration (override settings.py if needed)
NUM_EPISODES_PER_STRATEGY = 10
LOG_DIR = "logs"


@dataclass
class PriceDomainLogEntry:
    """Log entry for price domain experiments."""
    session_id: str
    timestamp: str
    model_name: str
    agent1_type: str
    agent2_type: str
    round_number: int
    round_duration: float
    turns: int
    buyer_max: float
    seller_min: float
    zopa_low: float
    zopa_high: float
    agreement: bool
    final_price: Optional[float]
    within_zopa: bool
    agent1_utility: float
    agent2_utility: float
    history_str: str
    json_details: str


class DeterministicCSVLogger:
    """CSV Logger for Janus vs Deterministic runs."""
    
    def __init__(self, session_name: str):
        self.session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.filename = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M')}_price_domain.csv"
        self.filepath = os.path.join(LOG_DIR, self.filename)
        
        os.makedirs(LOG_DIR, exist_ok=True)
        self.header_written = False
        
    def log_round(self, entry: PriceDomainLogEntry):
        """Write a round entry to CSV."""
        entry_dict = asdict(entry)
        file_exists = os.path.exists(self.filepath)
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(entry_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(self.filepath) == 0:
                writer.writeheader()
            writer.writerow(entry_dict)
    
    def get_filepath(self) -> str:
        return self.filepath


def get_all_strategies() -> List[str]:
    """Return list of all available deterministic strategies."""
    return list(STRATEGY_REGISTRY.keys())


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
    det_agent: DeterministicPriceAgent,
    janus_role: str,
    buyer_max: float,
    seller_min: float,
    starting_agent: str,
    episode_id: int,
    strategy_name: str,
    logger: DeterministicCSVLogger
) -> Dict[str, Any]:
    """
    Run a single negotiation episode between Janus and a deterministic agent.
    """
    start_time = time.time()
    
    domain = SingleIssuePriceDomain()
    domain.reset(episode_id)
    domain.buyer_max = buyer_max
    domain.seller_min = seller_min
    
    # Update Janus price bounds for normalization
    janus_agent.p_low = PRICE_RANGE_LOW
    janus_agent.p_high = PRICE_RANGE_HIGH
    
    history: List[Tuple[str, float]] = []
    conversation_history: List[Tuple[int, str]] = []
    last_offer: Optional[float] = None
    
    det_role = "seller" if janus_role == "buyer" else "buyer"
    det_reservation = seller_min if det_role == "seller" else buyer_max
    janus_reservation = buyer_max if janus_role == "buyer" else seller_min
    
    janus_agent_num = 1 if janus_role == "buyer" else 2
    det_agent_num = 2 if janus_role == "buyer" else 1
    
    current_agent = starting_agent
    
    agreement_reached = False
    final_price = None
    turn = 0
    
    print(f"\n{Fore.CYAN}--- Episode {episode_id} | Janus={janus_role} vs {strategy_name}={det_role} | Start={starting_agent} ---{Fore.RESET}")
    print(f"    ZOPA: Seller Min=${seller_min:.2f}, Buyer Max=${buyer_max:.2f}")
    
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
            print(f"{Fore.GREEN}  [T{turn}] Janus ({janus_role}): {response}{Fore.RESET}")
            
            action = domain.parse_agent_action(janus_agent_num, response)
            
            if action.action_type == "OFFER" and action.offer_content is not None:
                last_offer = action.offer_content
                history.append((janus_role, last_offer))
                conversation_history.append((janus_agent_num, f"OFFER {last_offer}"))
                domain.current_offer = last_offer
                domain.last_offerer_id = janus_agent_num
                
            elif action.action_type == "ACCEPT" and last_offer is not None:
                agreement_reached = True
                final_price = last_offer
                conversation_history.append((janus_agent_num, "ACCEPT"))
                print(f"{Fore.YELLOW}  [T{turn}] Janus ACCEPTS at ${final_price:.2f}{Fore.RESET}")
            
            current_agent = "deterministic"
            
        else:
            # Deterministic agent's turn
            state = create_price_state(
                timestep=turn,
                max_turns=MAX_TURNS,
                role=det_role,
                last_offer=last_offer,
                history=history,
                reservation=det_reservation,
                buyer_max=buyer_max,
                seller_min=seller_min
            )
            
            action = det_agent.propose_action(state)
            
            if action.type == "OFFER":
                last_offer = action.price
                history.append((det_role, last_offer))
                conversation_history.append((det_agent_num, f"OFFER {last_offer}"))
                domain.current_offer = last_offer
                domain.last_offerer_id = det_agent_num
                print(f"{Fore.BLUE}  [T{turn}] {strategy_name} ({det_role}): OFFER ${action.price:.2f}{Fore.RESET}")
                
            elif action.type == "ACCEPT" and last_offer is not None:
                agreement_reached = True
                final_price = last_offer
                conversation_history.append((det_agent_num, "ACCEPT"))
                print(f"{Fore.YELLOW}  [T{turn}] {strategy_name} ACCEPTS at ${final_price:.2f}{Fore.RESET}")
            
            current_agent = "janus"
    
    # Calculate utilities
    round_duration = time.time() - start_time
    zopa_width = buyer_max - seller_min
    
    if agreement_reached and final_price is not None:
        buyer_utility = buyer_max - final_price
        seller_utility = final_price - seller_min
        within_zopa = seller_min <= final_price <= buyer_max
        
        agent1_utility = buyer_utility
        agent2_utility = seller_utility
        
        janus_utility = buyer_utility if janus_role == "buyer" else seller_utility
        det_utility = seller_utility if janus_role == "buyer" else buyer_utility
        
        janus_norm = janus_utility / zopa_width if zopa_width > 0 else 0
        det_norm = det_utility / zopa_width if zopa_width > 0 else 0
        
        print(f"{Fore.CYAN}  AGREEMENT at ${final_price:.2f} | Janus util={janus_utility:.2f} ({janus_norm:.2%}) | {strategy_name} util={det_utility:.2f} ({det_norm:.2%}){Fore.RESET}")
    else:
        agent1_utility = 0.0
        agent2_utility = 0.0
        janus_utility = 0
        det_utility = 0
        janus_norm = 0
        det_norm = 0
        within_zopa = False
        print(f"{Fore.RED}  IMPASSE after {turn} turns{Fore.RESET}")
    
    # Build history string
    history_str = " | ".join([f"{agent_num}: {text}" for agent_num, text in conversation_history])
    
    agent1_type = "janus" if janus_role == "buyer" else strategy_name
    agent2_type = strategy_name if janus_role == "buyer" else "janus"
    
    outcome_details = {
        "agreement": agreement_reached,
        "agent1_utility": agent1_utility,
        "agent2_utility": agent2_utility,
        "price": final_price,
        "within_zopa": within_zopa
    }
    
    log_entry = PriceDomainLogEntry(
        session_id=logger.session_id,
        timestamp=datetime.now().isoformat(),
        model_name="janus_vs_" + strategy_name,
        agent1_type=agent1_type,
        agent2_type=agent2_type,
        round_number=episode_id,
        round_duration=round_duration,
        turns=turn,
        buyer_max=buyer_max,
        seller_min=seller_min,
        zopa_low=seller_min,
        zopa_high=buyer_max,
        agreement=agreement_reached,
        final_price=final_price,
        within_zopa=within_zopa,
        agent1_utility=agent1_utility,
        agent2_utility=agent2_utility,
        history_str=history_str,
        json_details=json.dumps(outcome_details)
    )
    
    logger.log_round(log_entry)
    
    return {
        "episode_id": episode_id,
        "strategy": strategy_name,
        "janus_role": janus_role,
        "starting_agent": starting_agent,
        "agreement": agreement_reached,
        "final_price": final_price,
        "turns": turn,
        "janus_utility": janus_utility,
        "janus_norm_utility": janus_norm,
        "det_utility": det_utility,
        "det_norm_utility": det_norm,
        "buyer_max": buyer_max,
        "seller_min": seller_min,
        "history": history
    }


async def run_against_strategy(
    strategy_name: str,
    num_episodes: int,
    logger: DeterministicCSVLogger,
    janus_buyer_rho: float = 0.2,
    janus_seller_rho: float = 0.8
) -> List[Dict[str, Any]]:
    """Run multiple episodes against a single strategy."""
    
    results = []
    
    # Create Janus agents (one for each role)
    janus_buyer = JanusAgent(
        agent_id=1,
        role="buyer",
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=janus_buyer_rho
    )
    
    janus_seller = JanusAgent(
        agent_id=2,
        role="seller", 
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=janus_seller_rho
    )
    
    for ep in range(1, num_episodes + 1):
        # Alternate Janus role
        janus_role = "buyer" if ep % 2 == 1 else "seller"
        janus_agent = janus_buyer if janus_role == "buyer" else janus_seller
        
        # Create deterministic opponent
        det_role = "seller" if janus_role == "buyer" else "buyer"
        det_agent = DeterministicPriceAgent(
            agent_id=2 if det_role == "seller" else 1,
            strategy_name=strategy_name
        )
        
        # Randomize starting agent
        starting_agent = random.choice(["janus", "deterministic"])
        
        # Generate ZOPA (matching training distribution)
        buyer_max = round(random.gauss(DEFAULT_BUYER_MAX_MEAN, DEFAULT_BUYER_MAX_STD), 2)
        seller_min = round(buyer_max - FIXED_ZOPA_WIDTH, 2)
        
        result = await run_episode(
            janus_agent=janus_agent,
            det_agent=det_agent,
            janus_role=janus_role,
            buyer_max=buyer_max,
            seller_min=seller_min,
            starting_agent=starting_agent,
            episode_id=ep,
            strategy_name=strategy_name,
            logger=logger
        )
        
        results.append(result)
    
    return results


def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary statistics."""
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}{Fore.RESET}\n")
    
    # Group by strategy
    by_strategy = {}
    for r in all_results:
        strat = r["strategy"]
        if strat not in by_strategy:
            by_strategy[strat] = []
        by_strategy[strat].append(r)
    
    print(f"{'Strategy':<25} {'Episodes':>8} {'Agreements':>10} {'Agr%':>8} {'Janus Norm':>12} {'Opp Norm':>12}")
    print("-" * 80)
    
    total_episodes = 0
    total_agreements = 0
    total_janus_norm = 0
    total_det_norm = 0
    
    for strat in sorted(by_strategy.keys()):
        results = by_strategy[strat]
        n = len(results)
        agreements = sum(1 for r in results if r["agreement"])
        janus_norm_sum = sum(r["janus_norm_utility"] for r in results if r["agreement"])
        det_norm_sum = sum(r["det_norm_utility"] for r in results if r["agreement"])
        
        agr_pct = agreements / n * 100 if n > 0 else 0
        janus_avg = janus_norm_sum / agreements if agreements > 0 else 0
        det_avg = det_norm_sum / agreements if agreements > 0 else 0
        
        print(f"{strat:<25} {n:>8} {agreements:>10} {agr_pct:>7.1f}% {janus_avg:>11.2%} {det_avg:>11.2%}")
        
        total_episodes += n
        total_agreements += agreements
        total_janus_norm += janus_norm_sum
        total_det_norm += det_norm_sum
    
    print("-" * 80)
    overall_agr_pct = total_agreements / total_episodes * 100 if total_episodes > 0 else 0
    overall_janus = total_janus_norm / total_agreements if total_agreements > 0 else 0
    overall_det = total_det_norm / total_agreements if total_agreements > 0 else 0
    print(f"{'OVERALL':<25} {total_episodes:>8} {total_agreements:>10} {overall_agr_pct:>7.1f}% {overall_janus:>11.2%} {overall_det:>11.2%}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Janus vs Deterministic evaluation")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES_PER_STRATEGY,
                       help="Number of episodes per strategy")
    parser.add_argument("--janus_buyer_rho", type=float, default=0.2,
                       help="Rho value for Janus as buyer")
    parser.add_argument("--janus_seller_rho", type=float, default=0.8,
                       help="Rho value for Janus as seller")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                       help="Specific strategies to test (default: all)")
    args = parser.parse_args()
    
    print(f"{Fore.MAGENTA}{'='*80}")
    print("JANUS vs DETERMINISTIC STRATEGIES EVALUATION")
    print(f"{'='*80}{Fore.RESET}\n")
    
    strategies = args.strategies if args.strategies else get_all_strategies()
    print(f"Strategies to test ({len(strategies)}): {strategies}\n")
    
    logger = DeterministicCSVLogger("janus_vs_deterministic")
    print(f"{Fore.GREEN}Logging to: {logger.get_filepath()}{Fore.RESET}\n")
    
    all_results = []
    global_episode_counter = 0
    
    for strategy_name in strategies:
        print(f"\n{Fore.MAGENTA}>>> Testing against: {strategy_name}{Fore.RESET}")
        
        try:
            results = await run_against_strategy(
                strategy_name=strategy_name,
                num_episodes=args.num_episodes,
                logger=logger,
                janus_buyer_rho=args.janus_buyer_rho,
                janus_seller_rho=args.janus_seller_rho
            )
            
            for r in results:
                global_episode_counter += 1
                r["global_episode_id"] = global_episode_counter
            
            all_results.extend(results)
        except Exception as e:
            print(f"{Fore.RED}Error with strategy {strategy_name}: {e}{Fore.RESET}")
            import traceback
            traceback.print_exc()
            continue
    
    print_summary(all_results)
    
    print(f"\n{Fore.GREEN}All results logged to: {logger.get_filepath()}{Fore.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
