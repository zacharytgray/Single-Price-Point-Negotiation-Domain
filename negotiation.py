"""
Single Price Point Negotiation Domain - Main Entry Point

This module provides the main negotiation loop for running single-issue
price negotiations between various agent types.

Usage:
    python negotiation.py --buyer_strategy boulware_conceding --seller_strategy linear_standard
    python negotiation.py --buyer_type janus --seller_strategy hardliner --janus_adapter checkpoints/final
    python negotiation.py --num_runs 100 --output logs/experiment.csv
"""

import asyncio
import random
import time
import argparse
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from colorama import Fore, init, Style

from config.settings import (
    DEFAULT_MODEL_NAME,
    MAX_TURNS,
    DEFAULT_BUYER_MAX_MEAN,
    DEFAULT_BUYER_MAX_STD,
    FIXED_ZOPA_WIDTH,
    PRICE_RANGE_LOW,
    PRICE_RANGE_HIGH
)
from src.core.price_structures import PriceAction, PriceState
from src.domain.single_issue_price_domain import SingleIssuePriceDomain
from src.agents.price_strategies import DeterministicPriceAgent, STRATEGY_REGISTRY
from src.agents.agent_factory import create_agent
from src.logging.csv_logger import CSVLogger
from src.logging.dataset_writer import DatasetWriter

# Initialize colorama
init(autoreset=True)


@dataclass 
class EpisodeResult:
    """Result of a single negotiation episode."""
    episode_id: str
    buyer_strategy: str
    seller_strategy: str
    buyer_type: str
    seller_type: str
    buyer_max: float
    seller_min: float
    zopa_width: float
    max_turns: int
    num_turns: int
    agreement: bool
    final_price: Optional[float]
    buyer_utility: Optional[float]
    seller_utility: Optional[float]
    rho: Optional[float]
    history: List[Tuple[str, float]]
    duration: float


class NegotiationEngine:
    """
    Main engine for running single-issue price negotiations.
    """
    
    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        seed: Optional[int] = None,
        csv_logger: Optional[CSVLogger] = None,
        dataset_writer: Optional[DatasetWriter] = None,
        verbose: bool = True
    ):
        """
        Initialize the negotiation engine.
        
        Args:
            max_turns: Maximum number of turns per episode
            seed: Random seed for reproducibility
            csv_logger: Optional CSV logger for results
            dataset_writer: Optional dataset writer for training data
            verbose: Whether to print progress
        """
        self.max_turns = max_turns
        self.verbose = verbose
        self.csv_logger = csv_logger
        self.dataset_writer = dataset_writer
        
        if seed is not None:
            random.seed(seed)
            
        self.domain = SingleIssuePriceDomain()
        
    def sample_zopa(self) -> Tuple[float, float]:
        """
        Sample buyer max and seller min for an episode.
        
        Returns:
            Tuple of (buyer_max, seller_min)
        """
        buyer_max = random.gauss(DEFAULT_BUYER_MAX_MEAN, DEFAULT_BUYER_MAX_STD)
        buyer_max = max(PRICE_RANGE_LOW + FIXED_ZOPA_WIDTH, min(PRICE_RANGE_HIGH, buyer_max))
        seller_min = buyer_max - FIXED_ZOPA_WIDTH
        return buyer_max, seller_min
    
    def run_episode(
        self,
        buyer_agent,
        seller_agent,
        buyer_max: Optional[float] = None,
        seller_min: Optional[float] = None,
        starting_role: str = "buyer",
        episode_id: Optional[str] = None
    ) -> EpisodeResult:
        """
        Run a single negotiation episode.
        
        Args:
            buyer_agent: Agent playing the buyer role
            seller_agent: Agent playing the seller role
            buyer_max: Buyer's maximum price (or sample if None)
            seller_min: Seller's minimum price (or sample if None)
            starting_role: Which role starts ("buyer" or "seller")
            episode_id: Unique ID for this episode
            
        Returns:
            EpisodeResult with negotiation outcome
        """
        start_time = time.time()
        
        # Generate episode ID if not provided
        if episode_id is None:
            episode_id = str(uuid.uuid4())[:8]
            
        # Sample ZOPA if not provided
        if buyer_max is None or seller_min is None:
            buyer_max, seller_min = self.sample_zopa()
            
        zopa_width = buyer_max - seller_min
        
        # Reset domain
        self.domain.reset(int(episode_id, 16) if len(episode_id) == 8 else hash(episode_id))
        self.domain.buyer_max = buyer_max
        self.domain.seller_min = seller_min
        
        # Initialize tracking
        history: List[Tuple[str, float]] = []
        last_offer: Optional[float] = None
        
        # Determine turn order
        current_role = starting_role
        
        agreement_reached = False
        final_price = None
        turn = 0
        
        if self.verbose:
            print(f"\n{Fore.CYAN}--- Episode {episode_id} ---{Fore.RESET}")
            print(f"    ZOPA: Seller Min=${seller_min:.2f}, Buyer Max=${buyer_max:.2f}")
            
        while turn < self.max_turns and not agreement_reached:
            turn += 1
            
            # Get current agent
            current_agent = buyer_agent if current_role == "buyer" else seller_agent
            reservation = buyer_max if current_role == "buyer" else seller_min
            
            # Build state
            state = PriceState(
                timestep=turn,
                max_turns=self.max_turns,
                role=current_role,
                last_offer_price=last_offer,
                offer_history=list(history),
                effective_reservation_price=reservation,
                true_reservation_price=reservation,
                public_price_range=(PRICE_RANGE_LOW, PRICE_RANGE_HIGH)
            )
            
            # Get action
            try:
                if asyncio.iscoroutinefunction(current_agent.propose_action):
                    action = asyncio.run(current_agent.propose_action(state))
                else:
                    action = current_agent.propose_action(state)
            except Exception as e:
                print(f"{Fore.RED}Error getting action: {e}{Fore.RESET}")
                action = PriceAction(type="OFFER", price=reservation)
                
            # Log step for dataset
            if self.dataset_writer:
                meta = {
                    "buyer_max": buyer_max,
                    "seller_min": seller_min,
                    "zopa_low": seller_min,
                    "zopa_high": buyer_max,
                    "zopa_width": zopa_width,
                    "accepted_price": None
                }
                agent_meta = {
                    "strategy": getattr(current_agent, 'strategy_name', 'unknown'),
                    "strategy_params": getattr(current_agent, 'params', {})
                }
                self.dataset_writer.add_step(
                    state=state,
                    action=action,
                    reward=0.0,
                    agent_metadata=agent_meta,
                    meta=meta,
                    trajectory_id=episode_id,
                    terminal=False
                )
                
            # Process action
            if action.type == "OFFER":
                if action.price is not None:
                    last_offer = action.price
                    history.append((current_role, action.price))
                    
                    if self.verbose:
                        color = Fore.BLUE if current_role == "buyer" else Fore.YELLOW
                        print(f"  Turn {turn}: {color}{current_role.upper()}{Fore.RESET} offers ${action.price:.2f}")
                        
            elif action.type == "ACCEPT":
                if last_offer is not None:
                    agreement_reached = True
                    final_price = last_offer
                    
                    if self.verbose:
                        color = Fore.BLUE if current_role == "buyer" else Fore.YELLOW
                        print(f"  Turn {turn}: {color}{current_role.upper()}{Fore.RESET} ACCEPTS ${final_price:.2f}")
                        
            # Switch roles
            current_role = "seller" if current_role == "buyer" else "buyer"
            
        # Calculate utilities
        buyer_utility = None
        seller_utility = None
        rho = None
        
        if agreement_reached and final_price is not None:
            buyer_utility = buyer_max - final_price
            seller_utility = final_price - seller_min
            if zopa_width > 0:
                # Outcome Rho - The actual normalized settlement point
                # 0.0 = Seller Min (Buyer wins everything)
                # 1.0 = Buyer Max (Seller wins everything)
                rho = (final_price - seller_min) / zopa_width
                
        duration = time.time() - start_time
        
        # Flush dataset writer
        if self.dataset_writer:
            if agreement_reached:
                outcome_meta = {
                    "agreement": True,
                    "price": final_price,
                    "turns": turn
                }
                self.dataset_writer.flush_episode(outcome_meta)
            else:
                outcome_meta = {
                    "agreement": False,
                    "price": None,
                    "turns": turn
                }
                self.dataset_writer.flush_episode_with_failure(outcome_meta)
                
        # Log to CSV
        if self.csv_logger:
            self.csv_logger.log_negotiation_result(
                run_id=episode_id,
                buyer_strategy=getattr(buyer_agent, 'strategy_name', 'unknown'),
                seller_strategy=getattr(seller_agent, 'strategy_name', 'unknown'),
                buyer_max=buyer_max,
                seller_min=seller_min,
                max_turns=self.max_turns,
                num_turns=turn,
                agreement=agreement_reached,
                final_price=final_price,
                buyer_type=getattr(buyer_agent, 'agent_type', 'deterministic'),
                seller_type=getattr(seller_agent, 'agent_type', 'deterministic')
            )
            
        if self.verbose:
            if agreement_reached:
                print(f"{Fore.GREEN}  Agreement at ${final_price:.2f} (Realized Rho={rho:.3f}){Fore.RESET}")
            else:
                print(f"{Fore.RED}  No agreement after {turn} turns{Fore.RESET}")
                
        return EpisodeResult(
            episode_id=episode_id,
            buyer_strategy=getattr(buyer_agent, 'strategy_name', 'unknown'),
            seller_strategy=getattr(seller_agent, 'strategy_name', 'unknown'),
            buyer_type=getattr(buyer_agent, 'agent_type', 'deterministic'),
            seller_type=getattr(seller_agent, 'agent_type', 'deterministic'),
            buyer_max=buyer_max,
            seller_min=seller_min,
            zopa_width=zopa_width,
            max_turns=self.max_turns,
            num_turns=turn,
            agreement=agreement_reached,
            final_price=final_price,
            buyer_utility=buyer_utility,
            seller_utility=seller_utility,
            rho=rho,
            history=history,
            duration=duration
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run single price point negotiations")
    
    # Agent configuration
    parser.add_argument("--buyer_strategy", type=str, default="boulware_conceding",
                       help="Buyer's strategy (for deterministic agents)")
    parser.add_argument("--seller_strategy", type=str, default="linear_standard",
                       help="Seller's strategy (for deterministic agents)")
    parser.add_argument("--buyer_type", type=str, default="deterministic",
                       choices=["deterministic", "llm", "janus", "basic", "basic_price", "price_strategy"],
                       help="Type of buyer agent")
    parser.add_argument("--seller_type", type=str, default="deterministic",
                       choices=["deterministic", "llm", "janus", "basic", "basic_price", "price_strategy"],
                       help="Type of seller agent")
    
    # Janus configuration
    parser.add_argument("--janus_adapter", type=str, default="checkpoints/final",
                       help="Path to Janus HyperLoRA adapter")
    parser.add_argument("--janus_model", type=str, default="Qwen/Qwen2-7B-Instruct",
                       help="Base model for Janus")
    parser.add_argument("--janus_rho", type=float, default=0.5,
                       help="Rho value for Janus agent")
    
    # LLM configuration
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                       help="Model name for LLM agents")
    
    # Episode configuration
    parser.add_argument("--num_runs", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS,
                       help="Maximum turns per episode")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    # Output configuration
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path")
    parser.add_argument("--dataset_out", type=str, default=None,
                       help="Output JSONL file for training data")
    parser.add_argument("--randomize_strategies", action="store_true",
                       help="Randomly select strategies for each episode (for dataset generation)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Auto-enable randomize_strategies for dataset generation with deterministic agents
    if args.dataset_out and args.buyer_type == "deterministic" and args.seller_type == "deterministic":
        if not args.randomize_strategies:
            print(f"{Fore.YELLOW}Note: Auto-enabling --randomize_strategies for diverse dataset generation{Fore.RESET}")
            args.randomize_strategies = True
    
    print(f"{Fore.CYAN}{'='*60}")
    print("Single Price Point Negotiation Domain")
    print(f"{'='*60}{Fore.RESET}")
    
    # Setup logging
    csv_logger = None
    if args.output:
        csv_logger = CSVLogger(args.output)
        print(f"Logging results to: {args.output}")
        
    dataset_writer = None
    if args.dataset_out:
        dataset_writer = DatasetWriter(args.dataset_out)
        print(f"Logging training data to: {args.dataset_out}")
        
    # Create engine
    engine = NegotiationEngine(
        max_turns=args.max_turns,
        seed=args.seed,
        csv_logger=csv_logger,
        dataset_writer=dataset_writer,
        verbose=not args.quiet
    )
    
    # Create agents
    print(f"\nBuyer: {args.buyer_type} ({args.buyer_strategy})")
    print(f"Seller: {args.seller_type} ({args.seller_strategy})")
    
    buyer_agent = create_agent(
        agent_type=args.buyer_type,
        role="buyer",
        strategy=args.buyer_strategy,
        model_name=args.model_name,
        janus_adapter_path=args.janus_adapter if args.buyer_type == "janus" else None,
        janus_model_path=args.janus_model if args.buyer_type == "janus" else None,
        rho=args.janus_rho if args.buyer_type == "janus" else None
    )
    
    seller_agent = create_agent(
        agent_type=args.seller_type,
        role="seller",
        strategy=args.seller_strategy,
        model_name=args.model_name,
        janus_adapter_path=args.janus_adapter if args.seller_type == "janus" else None,
        janus_model_path=args.janus_model if args.seller_type == "janus" else None,
        rho=args.janus_rho if args.seller_type == "janus" else None
    )
    
    # Run episodes
    results: List[EpisodeResult] = []
    
    print(f"\nRunning {args.num_runs} episodes...")
    if args.randomize_strategies:
        print("Using randomized strategies for dataset diversity")
    
    for i in range(args.num_runs):
        episode_id = f"{i:06x}"
        
        # Optionally randomize strategies for diversity
        if args.randomize_strategies:
            from src.agents.price_strategies import STRATEGY_REGISTRY
            buyer_strat = random.choice(list(STRATEGY_REGISTRY.keys()))
            seller_strat = random.choice(list(STRATEGY_REGISTRY.keys()))
            
            # Create new agents with random strategies
            episode_buyer = create_agent(
                agent_type="deterministic",
                role="buyer",
                strategy=buyer_strat,
                model_name=args.model_name,
                janus_adapter_path=None,
                janus_model_path=None,
                rho=None
            )
            episode_seller = create_agent(
                agent_type="deterministic",
                role="seller",
                strategy=seller_strat,
                model_name=args.model_name,
                janus_adapter_path=None,
                janus_model_path=None,
                rho=None
            )
        else:
            episode_buyer = buyer_agent
            episode_seller = seller_agent
        
        result = engine.run_episode(
            buyer_agent=episode_buyer,
            seller_agent=episode_seller,
            episode_id=episode_id
        )
        results.append(result)
        
    # Print summary
    agreements = sum(1 for r in results if r.agreement)
    avg_turns = sum(r.num_turns for r in results) / len(results)
    
    successful = [r for r in results if r.agreement and r.rho is not None]
    avg_rho = sum(r.rho for r in successful) / len(successful) if successful else 0.0
    
    print(f"\n{Fore.GREEN}{'='*60}")
    print("Summary")
    print(f"{'='*60}{Fore.RESET}")
    print(f"Episodes: {len(results)}")
    print(f"Agreements: {agreements} ({100*agreements/len(results):.1f}%)")
    print(f"Avg Turns: {avg_turns:.1f}")
    print(f"Avg Realized Rho: {avg_rho:.3f} (Target was {args.janus_rho if args.buyer_type == 'janus' or args.seller_type == 'janus' else 'N/A'})")
    
    if csv_logger:
        csv_logger.close()
        
    print("\nDone!")


if __name__ == "__main__":
    main()
