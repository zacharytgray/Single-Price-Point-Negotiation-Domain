"""
Visualize concession curves for all deterministic strategies.
Generates plots showing the offer trajectory over 20 turns.

Categories:
1. Independent Strategies: Plot Buyer and Seller curves on one chart.
2. Reactive Strategies: Plot Strategy vs Boulware opponent for both roles.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, ".")

from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.core.price_structures import PriceState, PriceAction

# Configuration
OUTPUT_DIR = Path("strategy_curve_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_TURNS = 20
BUYER_MAX = 1000.0  # Buyer Reservation (willing to pay up to 1000)
SELLER_MIN = 500.0  # Seller Reservation (willing to sell down to 500)
# ZOPA is [500, 1000]

OPPONENT_STRATEGY = "boulware_firm" # Standard opponent for reactive strategies

# Categorization
INDEPENDENT_STRATEGIES = [
    "boulware_very_conceding",
    "boulware_conceding",
    "boulware_firm",
    "boulware_hard",
    "linear_standard",
    "price_fixed_strict",
    "price_fixed_loose",
    "time_dependent",
    "hardliner",
    # Margin variants (start closer to reservation)
    "boulware_very_conceding_margin",
    "boulware_conceding_margin",
    "boulware_firm_margin",
    "boulware_hard_margin",
    # Bad strategy (immediately concedes to terrible position)
    "naive_concession"
]

REACTIVE_STRATEGIES = [
    "tit_for_tat",
    "split_difference",
    "micro_fine",
    "micro_moderate",
    "micro_coarse",
    "random_zopa" 
]


def simulate_independent_curve(strategy_name: str, role: str) -> list:
    """
    Simulate an independent strategy's offers over time.
    It doesn't depend on opponent offers, so we feed empty history.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        return []
        
    try:
        agent = DeterministicPriceAgent(1, strategy_name)
    except ValueError:
        return []
    
    # Parameters for simulation
    res_price = BUYER_MAX if role == "buyer" else SELLER_MIN
    
    # Note: Agents do NOT know opponent's reservation (ZOPA bounds)
    # They only know their own reservation and the public price range
    
    # Check if random_zopa needs ZOPA bounds (this strategy is special - uses known ZOPA)
    if strategy_name == "random_zopa":
        agent.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})

    offers = []
    offer_history = []
    
    for t in range(1, MAX_TURNS + 1):
        state = PriceState(
            timestep=t,
            max_turns=MAX_TURNS,
            role=role,
            last_offer_price=None, # Independent strategies do not need last offer
            offer_history=offer_history.copy(),
            effective_reservation_price=res_price,
            true_reservation_price=res_price,
            public_price_range=(0.0, 2000.0)
        )
        
        try:
            action = agent.propose_action(state)
            if action.type == "OFFER" and action.price is not None:
                offers.append(action.price)
                # Add to offer history for next iteration
                offer_history.append((role, action.price))
            else:
                # If it accepts (unlikely without input) or fails
                offers.append(None)
        except Exception as e:
            # print(f"Error simulating {strategy_name} ({role}) at t={t}: {e}")
            offers.append(None)
                
    return offers


def simulate_interaction(test_strat_name: str, test_role: str, opponent_strat_name: str):
    """
    Simulate a full interaction between strategy and opponent.
    Returns (strategy_offers, opponent_offers)
    """
    if test_strat_name not in STRATEGY_REGISTRY or opponent_strat_name not in STRATEGY_REGISTRY:
        return [], []

    # Setup agents
    if test_role == "buyer":
        buyer = DeterministicPriceAgent(1, test_strat_name)
        seller = DeterministicPriceAgent(2, opponent_strat_name)
        buyer_res = BUYER_MAX
        seller_res = SELLER_MIN
    else:
        buyer = DeterministicPriceAgent(1, opponent_strat_name)
        seller = DeterministicPriceAgent(2, test_strat_name)
        buyer_res = BUYER_MAX
        seller_res = SELLER_MIN

    # Provide ZOPA bounds to both agents so they start at ZOPA boundaries
    buyer.params.update({"zopa_low": SELLER_MIN, "zopa_high": BUYER_MAX})
    seller.params.update({"zopa_low": SELLER_MIN, "zopa_high": BUYER_MAX})

    # Handle random_zopa params
    if test_strat_name == "random_zopa":
        if test_role == "buyer": buyer.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})
        else: seller.params.update({"zopa_min": SELLER_MIN, "zopa_max": BUYER_MAX})
        
    history = []
    test_offers = []
    opp_offers = []
    last_price = None
    
    # To properly simulate reactive strategies, we need a back-and-forth.
    # We will simulate turns. 
    # Turn 1: Buyer offers.
    # Turn 2: Seller responds.
    # Turn 3: Buyer responds.
    # ...
    # This aligns with the "timestep" in the state usually incrementing per round or per turn.
    # In common negotiation loop: 
    # Round 1: Buyer Offer -> Seller Response
    # Let's align with that. 
    # BUT, 'propose_action' takes 'timestep'. Does timestep mean global turn or round?
    # Usually global turn index. 1, 2, 3...
    
    global_turn = 1
    agreement_reached = False
    
    while global_turn <= MAX_TURNS:
        # --- Buyer's Move ---
        if global_turn > MAX_TURNS: break
        
        b_state = PriceState(
            timestep=global_turn,
            max_turns=MAX_TURNS,
            role="buyer",
            last_offer_price=last_price,
            offer_history=list(history),
            effective_reservation_price=buyer_res,
            true_reservation_price=buyer_res,
            public_price_range=(0.0, 2000.0)
        )
        
        try:
            b_act = buyer.propose_action(b_state)
            if b_act.type == "OFFER":
                price = b_act.price
                history.append(("buyer", price))
                last_price = price
                
                if test_role == "buyer": test_offers.append((global_turn, price))
                else: opp_offers.append((global_turn, price))
            else: 
                # ACCEPT - but for visualization, we want to continue plotting
                # Record that we would accept, but continue with next offer
                agreement_reached = True
                # Continue to next turn to see what would happen
        except Exception:
            # If strategy crashes, just continue
            pass
            
        global_turn += 1
        
        # --- Seller's Move ---
        if global_turn > MAX_TURNS: break
        
        s_state = PriceState(
            timestep=global_turn,
            max_turns=MAX_TURNS,
            role="seller",
            last_offer_price=last_price,
            offer_history=list(history),
            effective_reservation_price=seller_res,
            true_reservation_price=seller_res,
            public_price_range=(0.0, 2000.0)
        )
        
        try:
            s_act = seller.propose_action(s_state)
            if s_act.type == "OFFER":
                price = s_act.price
                history.append(("seller", price))
                last_price = price
                
                if test_role == "seller": test_offers.append((global_turn, price))
                else: opp_offers.append((global_turn, price))
            else:
                # ACCEPT - but continue plotting for visualization
                agreement_reached = True
                # Continue to next turn
        except Exception:
            # If strategy crashes, just continue
            pass
            
        global_turn += 1
            
    return test_offers, opp_offers


def plot_independent(strategy_name):
    """Plot independent strategy curves for both roles."""
    buyer_offers = simulate_independent_curve(strategy_name, "buyer")
    seller_offers = simulate_independent_curve(strategy_name, "seller")
    
    if not buyer_offers and not seller_offers:
        return

    plt.figure(figsize=(10, 6))
    
    # Plot Buyer
    turns_b = range(1, len(buyer_offers) + 1)
    # Filter Nones
    b_points = [(t, p) for t, p in zip(turns_b, buyer_offers) if p is not None]
    if b_points:
        bx, by = zip(*b_points)
        plt.plot(bx, by, marker='o', label="As Buyer", color='blue', linewidth=2)
        
    # Plot Seller
    turns_s = range(1, len(seller_offers) + 1)
    s_points = [(t, p) for t, p in zip(turns_s, seller_offers) if p is not None]
    if s_points:
        sx, sy = zip(*s_points)
        plt.plot(sx, sy, marker='s', label="As Seller", color='red', linewidth=2)
        
    # Reference Lines
    plt.axhline(BUYER_MAX, color='blue', linestyle='--', alpha=0.3, label="Buyer Res (Max)")
    plt.axhline(SELLER_MIN, color='red', linestyle='--', alpha=0.3, label="Seller Res (Min)")
    plt.fill_between([1, MAX_TURNS], SELLER_MIN, BUYER_MAX, color='green', alpha=0.1, label="ZOPA")
    
    plt.title(f"Independent Strategy Curve: {strategy_name}")
    plt.xlabel("Turn")
    plt.ylabel("Price")
    plt.ylim(0, 1500) # Assuming broad range
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = OUTPUT_DIR / f"{strategy_name}_curve.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_reactive(strategy_name):
    """Plot reactive strategy playing against standard opponent."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Role: Buyer ---
    test_offers_b, opp_offers_s = simulate_interaction(strategy_name, "buyer", OPPONENT_STRATEGY)
    
    if test_offers_b:
        x, y = zip(*test_offers_b)
        ax1.plot(x, y, marker='o', color='blue', label=f"{strategy_name} (Buyer)", linewidth=2)
        
    if opp_offers_s:
        x, y = zip(*opp_offers_s)
        ax1.plot(x, y, marker='x', color='red', linestyle='--', label=f"{OPPONENT_STRATEGY} (Seller)", alpha=0.5)
        
    ax1.set_title(f"Role: Buyer vs {OPPONENT_STRATEGY}")
    ax1.set_ylim(0, 1500)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(BUYER_MAX, color='blue', linestyle=':', alpha=0.3, label="Buyer Res")
    ax1.axhline(SELLER_MIN, color='red', linestyle=':', alpha=0.3, label="Seller Res")
    ax1.fill_between([1, MAX_TURNS], SELLER_MIN, BUYER_MAX, color='green', alpha=0.1)
    ax1.legend()

    # --- Role: Seller ---
    test_offers_s, opp_offers_b = simulate_interaction(strategy_name, "seller", OPPONENT_STRATEGY)
    
    if test_offers_s:
        x, y = zip(*test_offers_s)
        ax2.plot(x, y, marker='s', color='red', label=f"{strategy_name} (Seller)", linewidth=2)
        
    if opp_offers_b:
        x, y = zip(*opp_offers_b)
        ax2.plot(x, y, marker='x', color='blue', linestyle='--', label=f"{OPPONENT_STRATEGY} (Buyer)", alpha=0.5)
        
    ax2.set_title(f"Role: Seller vs {OPPONENT_STRATEGY}")
    ax2.set_ylim(0, 1500)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(BUYER_MAX, color='blue', linestyle=':', alpha=0.3, label="Buyer Res")
    ax2.axhline(SELLER_MIN, color='red', linestyle=':', alpha=0.3, label="Seller Res")
    ax2.fill_between([1, MAX_TURNS], SELLER_MIN, BUYER_MAX, color='green', alpha=0.1)
    ax2.legend()
    
    plt.suptitle(f"Reactive Strategy Interaction: {strategy_name}", fontsize=14)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / f"{strategy_name}_interaction.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def main():
    print("Generating Strategy Curves...")
    
    for strat in INDEPENDENT_STRATEGIES:
        if strat in STRATEGY_REGISTRY:
            plot_independent(strat)
            
    for strat in REACTIVE_STRATEGIES:
        if strat in STRATEGY_REGISTRY:
            plot_reactive(strat)
            
    print("Done.")

if __name__ == "__main__":
    main()
