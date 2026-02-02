"""
Visualize concession curves for all deterministic strategies.

This script simulates each strategy against a stubborn opponent to reveal
the full concession pattern, then generates plots showing:
- Buyer concession curve (from low to reservation)
- Seller concession curve (from high to reservation)
- ZOPA region

Usage:
    python visualize_concessions.py
"""

import matplotlib.pyplot as plt
import os

from config.settings import MAX_TURNS, PRICE_RANGE_LOW, PRICE_RANGE_HIGH
from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.core.price_structures import PriceState

# Configuration
OUTPUT_DIR = "concession_plots"
BUYER_MAX = 900.0
SELLER_MIN = 400.0
# ZOPA is [400, 900]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def simulate_strategy(strategy_name: str):
    """
    Simulate a strategy playing both roles against stubborn opponents.
    
    Returns:
        Tuple of (buyer_offers, seller_offers) lists
    """
    # --- Simulate Buyer ---
    # Agent is Buyer (Max 900). Opponent acts as Stubborn Seller at 1500
    buyer_agent = DeterministicPriceAgent(1, strategy_name)
    
    # Inject Oracle params if needed
    if strategy_name == "random_zopa":
        buyer_agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})

    buyer_offers = []
    offer_history_buyer_view = []
    
    for t in range(1, MAX_TURNS + 1):
        state = PriceState(
            timestep=t,
            max_turns=MAX_TURNS,
            role="buyer",
            last_offer_price=None if t == 1 else 1500.0,
            offer_history=list(offer_history_buyer_view),
            effective_reservation_price=BUYER_MAX,
            true_reservation_price=BUYER_MAX,
            public_price_range=(PRICE_RANGE_LOW, PRICE_RANGE_HIGH)
        )
        
        try:
            action = buyer_agent.propose_action(state)
            if action.type == "OFFER":
                price = action.price
                buyer_offers.append(price)
                offer_history_buyer_view.append(("buyer", price))
                offer_history_buyer_view.append(("seller", 1500.0)) 
            else:
                buyer_offers.append(1500.0) 
        except Exception as e:
            print(f"Error in {strategy_name} (Buyer): {e}")
            buyer_offers.append(None)

    # --- Simulate Seller ---
    # Agent is Seller (Min 400). Opponent acts as Stubborn Buyer at 0
    seller_agent = DeterministicPriceAgent(2, strategy_name)
    if strategy_name == "random_zopa":
        seller_agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})

    seller_offers = []
    offer_history_seller_view = []
    
    for t in range(1, MAX_TURNS + 1):
        state = PriceState(
            timestep=t,
            max_turns=MAX_TURNS,
            role="seller",
            last_offer_price=None if t == 1 else 0.0,
            offer_history=list(offer_history_seller_view),
            effective_reservation_price=SELLER_MIN,
            true_reservation_price=SELLER_MIN,
            public_price_range=(PRICE_RANGE_LOW, PRICE_RANGE_HIGH)
        )
        
        try:
            action = seller_agent.propose_action(state)
            if action.type == "OFFER":
                price = action.price
                seller_offers.append(price)
                offer_history_seller_view.append(("seller", price))
                offer_history_seller_view.append(("buyer", 0.0))
            else:
                seller_offers.append(0.0)
        except Exception as e:
            print(f"Error in {strategy_name} (Seller): {e}")
            seller_offers.append(None)
            
    return buyer_offers, seller_offers


def plot_strategy(strategy_name: str, buyer_offers: list, seller_offers: list):
    """Generate and save concession plot for a strategy."""
    plt.figure(figsize=(10, 6))
    turns = range(1, len(buyer_offers) + 1)
    
    # Filter Nones
    safe_buyer_offers = [o if o is not None else 0 for o in buyer_offers]
    safe_seller_offers = [o if o is not None else 0 for o in seller_offers]
    
    plt.plot(turns, safe_buyer_offers, label=f"Buyer (Res={BUYER_MAX})", marker='o', color='blue')
    plt.plot(turns, safe_seller_offers, label=f"Seller (Res={SELLER_MIN})", marker='x', color='red')
    
    plt.axhline(y=BUYER_MAX, color='blue', linestyle='--', alpha=0.3, label="Buyer Max")
    plt.axhline(y=SELLER_MIN, color='red', linestyle='--', alpha=0.3, label="Seller Min")
    
    # Shade ZOPA
    plt.fill_between(turns, SELLER_MIN, BUYER_MAX, color='green', alpha=0.1, label="ZOPA")
    
    plt.title(f"Concession Curve: {strategy_name}")
    plt.xlabel("Turn")
    plt.ylabel("Price Offer")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(OUTPUT_DIR, f"concession_{strategy_name}.png")
    plt.savefig(filename)
    plt.close()
    
    return filename


def main():
    """Generate concession plots for all strategies."""
    strategies = list(STRATEGY_REGISTRY.keys())
    print(f"Found {len(strategies)} strategies: {strategies}")
    
    for strategy_name in strategies:
        print(f"Simulating {strategy_name}...")
        
        buyer_offers, seller_offers = simulate_strategy(strategy_name)
        filename = plot_strategy(strategy_name, buyer_offers, seller_offers)
        
    print(f"\nPlots saved to {os.path.abspath(OUTPUT_DIR)}")
    
    # Generate combined comparison plot
    generate_comparison_plot(strategies)


def generate_comparison_plot(strategies: list):
    """Generate a combined plot comparing all buyer strategies."""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(range(len(strategies)))
    
    for i, strategy_name in enumerate(strategies):
        buyer_agent = DeterministicPriceAgent(1, strategy_name)
        if strategy_name == "random_zopa":
            buyer_agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})
            
        offers = []
        history = []
        
        for t in range(1, MAX_TURNS + 1):
            state = PriceState(
                timestep=t,
                max_turns=MAX_TURNS,
                role="buyer",
                last_offer_price=None if t == 1 else 1500.0,
                offer_history=list(history),
                effective_reservation_price=BUYER_MAX,
                true_reservation_price=BUYER_MAX,
                public_price_range=(PRICE_RANGE_LOW, PRICE_RANGE_HIGH)
            )
            
            try:
                action = buyer_agent.propose_action(state)
                if action.type == "OFFER" and action.price is not None:
                    offers.append(action.price)
                    history.append(("buyer", action.price))
                    history.append(("seller", 1500.0))
                else:
                    offers.append(offers[-1] if offers else BUYER_MAX)
            except:
                offers.append(offers[-1] if offers else BUYER_MAX)
                
        turns = range(1, len(offers) + 1)
        plt.plot(turns, offers, label=strategy_name, color=colors[i], alpha=0.8)
        
    plt.axhline(y=BUYER_MAX, color='blue', linestyle='--', alpha=0.5, linewidth=2, label="Buyer Max")
    plt.fill_between(range(1, MAX_TURNS + 1), 0, BUYER_MAX, color='green', alpha=0.05)
    
    plt.title("Comparison of Buyer Concession Strategies")
    plt.xlabel("Turn")
    plt.ylabel("Price Offer")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, "comparison_all_strategies.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {filename}")


if __name__ == "__main__":
    main()
