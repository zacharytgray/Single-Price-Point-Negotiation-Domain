"""
Comprehensive Benchmark: Janus Agent vs All Deterministic Strategies

Runs Janus agent against every available deterministic strategy for 50 episodes each.
Logs all results and generates summary statistics.
"""

import asyncio
import random
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from colorama import Fore, Style, init

from src.agents.janus_agent import JanusAgent
from src.agents.price_strategies import DeterministicPriceAgent, STRATEGY_REGISTRY
from src.domain.single_issue_price_domain import SingleIssuePriceDomain
from src.core.price_structures import PriceState

init(autoreset=True)

# Configuration
JANUS_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
JANUS_ADAPTER_PATH = "checkpoints/final"
PRICE_RANGE_LOW = 200.0
PRICE_RANGE_HIGH = 1500.0
DEFAULT_BUYER_MAX_MEAN = 1000.0
DEFAULT_BUYER_MAX_STD = 200.0
FIXED_ZOPA_WIDTH = 500.0
MAX_TURNS = 20

# Benchmark Configuration
EPISODES_PER_STRATEGY = 50
JANUS_BUYER_RHO = 0.2
JANUS_SELLER_RHO = 0.8


class BenchmarkLogger:
    """CSV logger for benchmark results."""
    
    def __init__(self, output_path: str):
        self.filepath = output_path
        self.fieldnames = [
            "session_id", "timestamp", "strategy_name", "episode_id",
            "janus_role", "starting_agent", "round_duration", "turns",
            "buyer_max", "seller_min", "zopa_low", "zopa_high",
            "agreement", "final_price", "within_zopa",
            "janus_utility", "janus_norm_utility",
            "det_utility", "det_norm_utility",
            "history_str", "json_details"
        ]
        
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_episode(self, entry: Dict[str, Any]):
        """Log a single episode result."""
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(entry)


async def run_episode(
    janus_agent: JanusAgent,
    det_agent: DeterministicPriceAgent,
    janus_role: str,
    buyer_max: float,
    seller_min: float,
    starting_agent: str,
    episode_id: int,
    strategy_name: str
) -> Dict[str, Any]:
    """Run a single negotiation episode."""
    start_time = time.time()
    domain = SingleIssuePriceDomain()
    
    # Update Janus price bounds for normalization
    janus_agent.p_low = PRICE_RANGE_LOW
    janus_agent.p_high = PRICE_RANGE_HIGH
    
    # Initialize domain context
    janus_agent.domain_private_context = {
        "turn": 1,
        "max_turns": MAX_TURNS,
        "history": [],
        "max_willingness_to_pay": buyer_max if janus_role == "buyer" else None,
        "min_acceptable_price": seller_min if janus_role == "seller" else None,
    }
    
    history = []
    conversation_history = []
    agreement_reached = False
    final_price = None
    turn = 0
    
    # Determine starting agent
    current_agent = "janus" if starting_agent == "janus" else "deterministic"
    
    for turn in range(1, MAX_TURNS + 1):
        janus_agent.domain_private_context["turn"] = turn
        janus_agent.domain_private_context["history"] = history.copy()
        
        if current_agent == "janus":
            # Janus makes offer or accepts
            response = await janus_agent.generate_response()
            
            if "ACCEPT" in response.upper():
                agreement_reached = True
                if history:
                    final_price = history[-1][1]
                conversation_history.append((1 if janus_role == "buyer" else 2, "ACCEPT"))
                break
            
            # Extract price from response
            price_match = None
            import re
            for match in re.finditer(r'\d+\.?\d*', response):
                price_match = float(match.group())
            
            if price_match is None:
                # Fallback: maybe it said something else
                break
            
            # JanusAgent.generate_response() already denormalizes the price.
            real_price = round(price_match, 2)
            
            history.append((janus_role, real_price))
            conversation_history.append((1 if janus_role == "buyer" else 2, f"OFFER {real_price:.2f}"))
            current_agent = "deterministic"
            
        else:
            # Deterministic agent makes offer or accepts
            det_role = "seller" if janus_role == "buyer" else "buyer"
            det_res = seller_min if det_role == "seller" else buyer_max
            
            det_state = create_price_state(
                turn, MAX_TURNS, det_role,
                history[-1][1] if history else None,
                history, det_res, buyer_max, seller_min
            )
            
            det_action = det_agent.propose_action(det_state)
            
            if det_action.type == "ACCEPT":
                agreement_reached = True
                if history:
                    final_price = history[-1][1]
                conversation_history.append((2 if janus_role == "buyer" else 1, "ACCEPT"))
                break
            
            history.append((det_role, det_action.price))
            conversation_history.append((2 if janus_role == "buyer" else 1, f"OFFER {det_action.price:.2f}"))
            current_agent = "janus"
    
    round_duration = time.time() - start_time
    
    # Calculate utilities
    if agreement_reached and final_price is not None:
        buyer_util = buyer_max - final_price
        seller_util = final_price - seller_min
        within_zopa = seller_min <= final_price <= buyer_max
    else:
        buyer_util = 0.0
        seller_util = 0.0
        within_zopa = False
    
    janus_utility = buyer_util if janus_role == "buyer" else seller_util
    det_utility = seller_util if janus_role == "buyer" else buyer_util
    
    zopa_size = buyer_max - seller_min
    janus_norm = janus_utility / zopa_size if zopa_size > 0 else 0.0
    det_norm = det_utility / zopa_size if zopa_size > 0 else 0.0
    
    history_str = " | ".join([f"{num}: {text}" for num, text in conversation_history])
    
    return {
        "episode_id": episode_id,
        "strategy_name": strategy_name,
        "janus_role": janus_role,
        "starting_agent": starting_agent,
        "round_duration": round_duration,
        "turns": turn,
        "buyer_max": buyer_max,
        "seller_min": seller_min,
        "zopa_low": seller_min,
        "zopa_high": buyer_max,
        "agreement": agreement_reached,
        "final_price": final_price,
        "within_zopa": within_zopa,
        "janus_utility": janus_utility,
        "janus_norm_utility": janus_norm,
        "det_utility": det_utility,
        "det_norm_utility": det_norm,
        "history": history,
        "history_str": history_str,
        "json_details": json.dumps({
            "agreement": agreement_reached,
            "janus_utility": janus_utility,
            "det_utility": det_utility,
            "price": final_price,
            "within_zopa": within_zopa
        })
    }


def create_price_state(timestep, max_turns, role, last_offer, history, reservation, buyer_max, seller_min):
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


async def benchmark_strategy(
    strategy_name: str,
    num_episodes: int,
    logger: BenchmarkLogger,
    session_id: str
) -> Dict[str, Any]:
    """Run benchmark against a single strategy."""
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"Strategy: {Fore.YELLOW}{strategy_name}{Fore.CYAN} ({num_episodes} episodes)")
    print(f"{'='*80}{Fore.RESET}")
    
    # Create Janus agents
    janus_buyer = JanusAgent(
        agent_id=1, role="buyer",
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=JANUS_BUYER_RHO
    )
    
    janus_seller = JanusAgent(
        agent_id=2, role="seller",
        model_path=JANUS_MODEL_PATH,
        adapter_path=JANUS_ADAPTER_PATH,
        rho=JANUS_SELLER_RHO
    )
    
    results = []
    agreements = 0
    total_janus_utility = 0.0
    total_det_utility = 0.0
    
    for ep in range(1, num_episodes + 1):
        # Alternate Janus role: odd episodes = buyer, even episodes = seller
        janus_role = "buyer" if ep % 2 == 1 else "seller"
        janus_agent = janus_buyer if janus_role == "buyer" else janus_seller
        
        det_role = "seller" if janus_role == "buyer" else "buyer"
        det_agent = DeterministicPriceAgent(
            agent_id=2 if det_role == "seller" else 1,
            strategy_name=strategy_name
        )
        
        # Ensure exactly half start with Janus, half with deterministic
        # Episodes 1, 3, 5, ... (odd) -> Janus starts
        # Episodes 2, 4, 6, ... (even) -> Deterministic starts
        starting_agent = "janus" if ep % 2 == 1 else "deterministic"
        
        buyer_max = round(random.gauss(DEFAULT_BUYER_MAX_MEAN, DEFAULT_BUYER_MAX_STD), 2)
        seller_min = round(buyer_max - FIXED_ZOPA_WIDTH, 2)
        
        result = await run_episode(
            janus_agent, det_agent, janus_role,
            buyer_max, seller_min, starting_agent,
            ep, strategy_name
        )
        
        # Log to CSV
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            **{k: v for k, v in result.items() if k != "history"}
        }
        logger.log_episode(log_entry)
        
        results.append(result)
        
        if result["agreement"]:
            agreements += 1
            total_janus_utility += result["janus_utility"]
            total_det_utility += result["det_utility"]
        
        # Progress output
        status = f"{Fore.GREEN}✓" if result["agreement"] else f"{Fore.RED}✗"
        price_display = f"{result['final_price']:.2f}" if result['final_price'] else "N/A"
        print(f"  Ep {ep:2d}/{num_episodes}: {status} "
              f"Janus={janus_role[0].upper()} Start={starting_agent[0].upper()} "
              f"Turns={result['turns']:2d} "
              f"Price={price_display:>7s}{Fore.RESET}")
    
    # Summary stats
    agreement_rate = agreements / num_episodes
    avg_janus_util = total_janus_utility / agreements if agreements > 0 else 0.0
    avg_det_util = total_det_utility / agreements if agreements > 0 else 0.0
    
    print(f"\n{Fore.CYAN}Summary for {strategy_name}:")
    print(f"  Agreement Rate: {Fore.YELLOW}{agreement_rate*100:.1f}%{Fore.RESET}")
    print(f"  Avg Janus Utility: {Fore.YELLOW}{avg_janus_util:.2f}{Fore.RESET}")
    print(f"  Avg Det Utility: {Fore.YELLOW}{avg_det_util:.2f}{Fore.RESET}")
    
    return {
        "strategy": strategy_name,
        "episodes": num_episodes,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "avg_janus_utility": avg_janus_util,
        "avg_det_utility": avg_det_util
    }


async def run_full_benchmark():
    """Run full benchmark against all strategies."""
    
    session_id = f"janus_full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = f"logs/{session_id}.csv"
    
    Path("logs").mkdir(exist_ok=True)
    logger = BenchmarkLogger(log_path)
    
    # Get all strategies except random_zopa (oracle strategy) and completed ones
    excluded_strategies = ["random_zopa"]
    all_strategies = [name for name in STRATEGY_REGISTRY.keys() if name not in excluded_strategies]
    
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.GREEN}JANUS AGENT FULL BENCHMARK")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"Total Strategies: {Fore.YELLOW}{len(all_strategies)}{Fore.RESET}")
    print(f"Episodes per Strategy: {Fore.YELLOW}{EPISODES_PER_STRATEGY}{Fore.RESET}")
    print(f"Total Episodes: {Fore.YELLOW}{len(all_strategies) * EPISODES_PER_STRATEGY}{Fore.RESET}")
    print(f"Janus Buyer Rho: {Fore.YELLOW}{JANUS_BUYER_RHO}{Fore.RESET}")
    print(f"Janus Seller Rho: {Fore.YELLOW}{JANUS_SELLER_RHO}{Fore.RESET}")
    print(f"Starting Agent: {Fore.YELLOW}Alternates (odd=Janus, even=Det){Fore.RESET}")
    print(f"Log File: {Fore.YELLOW}{log_path}{Fore.RESET}")
    print(f"{Fore.CYAN}{'='*80}{Fore.RESET}\n")
    
    start_time = time.time()
    strategy_summaries = []
    
    for idx, strategy in enumerate(all_strategies, 1):
        print(f"\n{Fore.MAGENTA}[{idx}/{len(all_strategies)}]{Fore.RESET} ", end="")
        summary = await benchmark_strategy(strategy, EPISODES_PER_STRATEGY, logger, session_id)
        strategy_summaries.append(summary)
        
        elapsed = time.time() - start_time
        avg_per_strategy = elapsed / idx
        remaining = avg_per_strategy * (len(all_strategies) - idx)
        print(f"{Fore.CYAN}  Elapsed: {elapsed/60:.1f}m | Est. Remaining: {remaining/60:.1f}m{Fore.RESET}")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n\n{Fore.GREEN}{'='*80}")
    print(f"BENCHMARK COMPLETE!")
    print(f"{'='*80}{Fore.RESET}")
    print(f"Total Time: {Fore.YELLOW}{total_time/60:.1f} minutes{Fore.RESET}")
    print(f"Results saved to: {Fore.YELLOW}{log_path}{Fore.RESET}\n")
    
    # Overall statistics
    print(f"{Fore.CYAN}Overall Performance Summary:{Fore.RESET}")
    print(f"{'Strategy':<30} {'Agreement %':>12} {'Avg Janus U':>12} {'Avg Det U':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    
    for s in strategy_summaries:
        print(f"{s['strategy']:<30} {s['agreement_rate']*100:>11.1f}% "
              f"{s['avg_janus_utility']:>12.2f} {s['avg_det_utility']:>12.2f}")
    
    # Save summary to JSON
    summary_path = f"logs/{session_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "session_id": session_id,
            "total_time_seconds": total_time,
            "total_strategies": len(all_strategies),
            "episodes_per_strategy": EPISODES_PER_STRATEGY,
            "janus_buyer_rho": JANUS_BUYER_RHO,
            "janus_seller_rho": JANUS_SELLER_RHO,
            "strategy_summaries": strategy_summaries
        }, f, indent=2)
    
    print(f"\n{Fore.GREEN}Summary saved to: {Fore.YELLOW}{summary_path}{Fore.RESET}")


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())
