"""
Strategy registry and implementations for deterministic price negotiation agents.
"""
import random
import math
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List
from src.core.price_structures import PriceAction, PriceState
from src.agents.base_agent import BaseAgent

# --- Strategy Logic Implementations ---

def strategy_boulware(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Boulware strategy: time-dependent concession curve.
    Target = Start + (Reservation - Start) * (t/T)^beta
    
    Starts at opponent's reservation (ZOPA boundary) and concedes toward own reservation.
    If static_margin is provided, starts at Reservation +/- margin instead.
    """
    beta = params.get("beta", 1.0)
    reservation = state.effective_reservation_price
    
    # Cap time_frac to prevent full capitulation
    concession_cap = params.get("concession_cap", 0.95)
    time_frac = min(concession_cap, state.timestep / state.max_turns)
    
    # Get public price range
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate symmetric starting positions around ZOPA
    # Buyer at reservation R starts at R - ZOPA_WIDTH (concedes ZOPA_WIDTH units)
    # Seller at reservation R starts at R + ZOPA_WIDTH (concedes ZOPA_WIDTH units)
    # This ensures both agents concede the same distance to reach their reservation
    DEFAULT_ZOPA_WIDTH = 500.0  # Standard ZOPA width from paper
    margin = params.get("initial_margin", DEFAULT_ZOPA_WIDTH)
    
    if state.role == "buyer":
        # Buyer: Start at reservation - margin, concede UP to own reservation
        start_price = max(pub_min, reservation - margin)
        target_price = start_price + (reservation - start_price) * (time_frac ** beta)
        target_price = min(target_price, reservation)
        
    else:  # seller
        # Seller: Start at reservation + margin, concede DOWN to own reservation
        start_price = min(pub_max, reservation + margin)
        target_price = start_price + (reservation - start_price) * (time_frac ** beta)
        target_price = max(target_price, reservation)

    # Check for acceptance
    if state.last_offer_price is not None:
        if state.role == "buyer":
            if state.last_offer_price <= target_price:
                return PriceAction(type="ACCEPT", price=None)
            if state.last_offer_price <= reservation and state.last_offer_price <= target_price:
                return PriceAction(type="ACCEPT", price=None)
        else:  # seller
            if state.last_offer_price >= target_price:
                return PriceAction(type="ACCEPT", price=None)
            if state.last_offer_price >= reservation and state.last_offer_price >= target_price:
                return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(target_price, 2))


def strategy_noisy_boulware(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Noisy Boulware strategy: Boulware concession with bounded randomness.
    
    Uses the base boulware calculation but adds slight, bounded random noise to offers.
    This keeps the opponent guessing while maintaining the overall concession pattern.
    
    Parameters:
        - beta: Concession curve parameter (same as boulware)
        - noise_max: Maximum absolute noise to add (default: 25.0)
        - noise_scale: Scale of noise relative to ZOPA width (default: 0.05 = 5%)
    
    The noise is bounded to ensure the agent doesn't accidentally make "better" offers
    than intended (e.g., a buyer offering more than the noisy target).
    """
    # Get base boulware target
    base_result = strategy_boulware(state, params)
    
    # If accepting, return as-is
    if base_result.type == "ACCEPT":
        return base_result
    
    base_price = base_result.price
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate noise bounds
    DEFAULT_ZOPA_WIDTH = 500.0
    noise_max = params.get("noise_max", 25.0)
    noise_scale = params.get("noise_scale", 0.05)
    zopa_based_noise = DEFAULT_ZOPA_WIDTH * noise_scale
    actual_noise_max = min(noise_max, zopa_based_noise)
    
    # Generate bounded random noise
    # For buyers: can offer LESS (lower price is better for buyer) but not MORE than base
    # For sellers: can offer MORE (higher price is better for seller) but not LESS than base
    # This ensures noise doesn't hurt the agent's position
    noise = random.uniform(-actual_noise_max, actual_noise_max)
    
    if role == "buyer":
        # Buyer: base_price is what they're willing to pay (upper bound)
        # Noise can make them offer slightly less (good) or slightly more (bad, bounded)
        # Allow going slightly above base (worse for buyer) or below (better for buyer)
        noisy_price = base_price + noise
        # Clamp: can't offer more than base + noise_max (worse position)
        # Can't offer below pub_min (invalid)
        noisy_price = min(noisy_price, base_price + actual_noise_max)
        noisy_price = max(noisy_price, pub_min)
        # Also can't exceed reservation (would be irrational)
        noisy_price = min(noisy_price, reservation)
    else:
        # Seller: base_price is what they're willing to accept (lower bound)
        # Noise can make them ask slightly more (good) or slightly less (bad, bounded)
        noisy_price = base_price + noise
        # Clamp: can't ask less than base - noise_max (worse position)
        # Can't ask above pub_max (invalid)
        noisy_price = max(noisy_price, base_price - actual_noise_max)
        noisy_price = min(noisy_price, pub_max)
        # Also can't go below reservation (would be irrational)
        noisy_price = max(noisy_price, reservation)
    
    return PriceAction(type="OFFER", price=round(noisy_price, 2))


def strategy_price_fixed(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Fixed Price strategy: Always offers a fixed position inside the ZOPA.
    
    Buyer offers LOW (just above seller's reservation + margin) to get good deals.
    Seller offers HIGH (just below buyer's reservation - margin) to get good deals.
    
    Uses ZOPA bounds if available, otherwise estimates from public price range.
    """
    margin = params.get("margin", 0.0)
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Estimate opponent's reservation from ZOPA bounds if available, else from public range
    # Standard ZOPA width is 500, so opponent's reservation is 500 away
    DEFAULT_ZOPA_WIDTH = 500.0
    
    if state.role == "buyer":
        # Buyer estimates seller's reservation (buyer's max - 500)
        # and offers LOW but INSIDE ZOPA: seller_res + margin
        seller_reservation = reservation - DEFAULT_ZOPA_WIDTH
        target_price = seller_reservation + margin
        target_price = max(seller_reservation, target_price)  # Stay at or above seller's min
        
        if state.last_offer_price is not None and state.last_offer_price <= target_price:
            return PriceAction(type="ACCEPT", price=None)
    else:
        # Seller estimates buyer's reservation (seller's min + 500)
        # and offers HIGH but INSIDE ZOPA: buyer_res - margin
        buyer_reservation = reservation + DEFAULT_ZOPA_WIDTH
        target_price = buyer_reservation - margin
        target_price = min(buyer_reservation, target_price)  # Stay at or below buyer's max
        
        if state.last_offer_price is not None and state.last_offer_price >= target_price:
            return PriceAction(type="ACCEPT", price=None)
            
    return PriceAction(type="OFFER", price=round(target_price, 2))


def strategy_tit_for_tat(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Tit-for-Tat: Mirror opponent's concession.
    Start at opponent's reservation (ZOPA boundary) by default.
    """
    role = state.role
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate symmetric starting positions around ZOPA
    # Both agents start ZOPA_WIDTH away from their reservation
    DEFAULT_ZOPA_WIDTH = 500.0
    margin = params.get("initial_margin", DEFAULT_ZOPA_WIDTH)
    
    # Initial offer or if no history
    if not state.offer_history:
        if role == "buyer":
            start_price = max(pub_min, reservation - margin)
            return PriceAction(type="OFFER", price=round(start_price, 2))
        else:
            start_price = min(pub_max, reservation + margin)
            return PriceAction(type="OFFER", price=round(start_price, 2))
            
    # Filter history
    my_offers = [p for r, p in state.offer_history if r == role]
    opp_offers = [p for r, p in state.offer_history if r != role]
    
    if not my_offers:
        # Second turn - use same logic as first turn
        if role == "buyer":
            target = max(pub_min, reservation - margin)
        else:
            target = min(pub_max, reservation + margin)
         
        if state.last_offer_price is not None:
            if role == "buyer" and state.last_offer_price <= target:
                return PriceAction(type="ACCEPT", price=None)
            if role == "seller" and state.last_offer_price >= target:
                return PriceAction(type="ACCEPT", price=None)
        return PriceAction(type="OFFER", price=round(target, 2))
    
    if len(opp_offers) < 2:
        target = my_offers[-1]
    else:
        concession = abs(opp_offers[-1] - opp_offers[-2])
        
        if role == "buyer":
            target = my_offers[-1] + concession
            target = min(target, reservation)
        else:
            target = my_offers[-1] - concession
            target = max(target, reservation)
            
    # Check acceptance
    if state.last_offer_price is not None:
        is_acceptable = (state.last_offer_price <= reservation if role == "buyer" else state.last_offer_price >= reservation)
        is_better_than_target = (state.last_offer_price <= target if role == "buyer" else state.last_offer_price >= target)
        
        if is_acceptable and is_better_than_target:
            return PriceAction(type="ACCEPT", price=None)
             
    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_linear(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Linear concessions from Start to Reservation over MaxRounds.
    """
    return strategy_boulware(state, {**params, "beta": 1.0})


def strategy_split_difference(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Split difference between last offer and own previous offer (or reservation).
    If initial_margin is None, starts at opponent's reservation (ZOPA boundary).
    """
    role = state.role
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Get ZOPA bounds - default to public bounds if not provided
    zopa_low = params.get("zopa_low", pub_min)
    zopa_high = params.get("zopa_high", pub_max)
    
    if state.last_offer_price is None:
        margin = params.get("initial_margin", None)
        if margin is not None:
            target = reservation - margin if role == "buyer" else reservation + margin
        else:
            # Start at opponent's reservation (ZOPA boundary)
            target = zopa_low if role == "buyer" else zopa_high
        return PriceAction(type="OFFER", price=round(target, 2))
        
    opp_price = state.last_offer_price
    
    my_offers = [p for r, p in state.offer_history if r == role]
    if not my_offers:
        my_anchor = reservation
    else:
        my_anchor = my_offers[-1]
        
    midpoint = (my_anchor + opp_price) / 2.0
    
    if role == "buyer":
        target = min(midpoint, reservation)
        if opp_price <= target:
            return PriceAction(type="ACCEPT", price=None)
    else:
        target = max(midpoint, reservation)
        if opp_price >= target:
            return PriceAction(type="ACCEPT", price=None)
             
    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_time_dependent_threshold(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Accepts only if offer improves. Threshold relaxes as deadline approaches.
    Starts symmetrically around ZOPA (ZOPA_WIDTH away from reservation).
    """
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate symmetric starting positions around ZOPA
    DEFAULT_ZOPA_WIDTH = 500.0
    margin = params.get("margin", DEFAULT_ZOPA_WIDTH)
    
    t = state.timestep
    T = state.max_turns
    
    concession_cap = params.get("concession_cap", 0.95)
    frac = min(concession_cap, t / T)
    
    if role == "buyer":
        # Start at reservation - margin, concede to reservation
        start_price = max(pub_min, reservation - margin)
            
        current_threshold = start_price + (reservation - start_price) * frac
        target = min(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price <= target:
            return PriceAction(type="ACCEPT", price=None)
    else:
        # Start at reservation + margin, concede to reservation
        start_price = min(pub_max, reservation + margin)
            
        current_threshold = start_price - (start_price - reservation) * frac
        target = max(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price >= target:
            return PriceAction(type="ACCEPT", price=None)
            
    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_hardliner(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Maintains a tough stance until the very last round, then concedes to reservation.
    If margin is None, holds at opponent's reservation (ZOPA boundary).
    """
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate symmetric hold positions around ZOPA
    DEFAULT_ZOPA_WIDTH = 500.0
    margin = params.get("margin", DEFAULT_ZOPA_WIDTH)
    
    # Determine hold position (symmetric around ZOPA)
    if role == "buyer":
        hold_price = max(pub_min, reservation - margin)
    else:
        hold_price = min(pub_max, reservation + margin)
    
    # Check if we should cave in (Last Round)
    if state.timestep >= state.max_turns - 1:
        cave_in_margin = params.get("cave_in_margin", 5.0)
        if role == "buyer":
            target = max(pub_min, reservation - cave_in_margin)
        else:
            target = min(pub_max, reservation + cave_in_margin)
    else:
        target = hold_price
            
    if state.last_offer_price is not None:
        if role == "buyer" and state.last_offer_price <= target:
            return PriceAction(type="ACCEPT", price=None)
        if role == "seller" and state.last_offer_price >= target:
            return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_random_in_zopa(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Randomly offers within ZOPA. Oracle strategy (training only).
    Requires ZOPA bounds passed in params.
    """
    zopa_min = params.get("zopa_min")
    zopa_max = params.get("zopa_max")
    
    if zopa_min is None or zopa_max is None:
        return strategy_linear(state, {})
        
    offer = random.uniform(zopa_min, zopa_max)
    
    if state.last_offer_price is not None:
        if zopa_min <= state.last_offer_price <= zopa_max:
            if random.random() < 0.3:
                return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(offer, 2))


def strategy_naive_boulware(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Naive Boulware strategy - concedes in the WRONG direction with hardline-then-sudden-drop curve.

    A normal agent starts far from their reservation and concedes TOWARD it over time.
    This agent does the opposite: starts AT their reservation and moves AWAY from it,
    making progressively worse offers as the deadline approaches.

    Buyer: starts at reservation (high), drifts DOWN — offering less and less over time.
    Seller: starts at reservation (low), drifts UP — asking more and more over time.

    Uses beta=3.0 (hardline Boulware) so it stays near reservation for most of the negotiation,
    then concedes rapidly toward the end. The curve is convex — flat early, steep late.
    """
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)

    DEFAULT_ZOPA_WIDTH = 500.0
    beta = params.get("beta", 3.0)  # Hardline Boulware: flat early, steep late
    concession_cap = params.get("concession_cap", 0.95)
    time_frac = min(concession_cap, state.timestep / state.max_turns)
    # With beta > 1: (t/T)^beta grows slowly at first, then rapidly near the end
    drift = DEFAULT_ZOPA_WIDTH * (time_frac ** beta)

    if role == "buyer":
        # Buyer starts at reservation and drifts DOWN (worse offers over time)
        target_price = max(pub_min, reservation - drift)

        # Accept if opponent offers something at or below reservation (we're naive but not irrational)
        if state.last_offer_price is not None and state.last_offer_price <= reservation:
            return PriceAction(type="ACCEPT", price=None)
    else:
        # Seller starts at reservation and drifts UP (worse offers over time)
        target_price = min(pub_max, reservation + drift)

        # Accept if opponent offers something at or above reservation
        if state.last_offer_price is not None and state.last_offer_price >= reservation:
            return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(target_price, 2))


def strategy_naive_concession(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Naive Concession strategy - consistently makes bad offers with slight random variation.
    
    This agent doesn't understand negotiation dynamics and simply makes offers
    that are terrible for itself (but great for the opponent).
    
    Buyer: Offers near its reservation (high prices) - pays way too much
    Seller: Offers near its reservation (low prices) - earns way too little
    
    Adds small random jitter to vary offers slightly, but never makes "good" offers.
    """
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Calculate the "bad" offer zone - near own reservation
    # Use a small buffer so we don't accidentally offer beyond reservation
    buffer = params.get("buffer", 25.0)  # Stay within $25 of own reservation
    jitter = params.get("jitter", 15.0)  # Random variation up to $15
    
    # Generate base "bad" offer near own reservation
    if role == "buyer":
        # Bad for buyer = high price near reservation
        base_price = reservation - buffer  # Close to max willing to pay
        # Add random jitter (could go slightly higher or lower, but always bad)
        variation = random.uniform(-jitter, jitter)
        naive_price = base_price + variation
        # Clamp to valid range
        naive_price = max(pub_min, min(naive_price, reservation))
        
        # Accept if opponent offers something reasonable (we're naive!)
        if state.last_offer_price is not None and state.last_offer_price <= reservation:
            return PriceAction(type="ACCEPT", price=None)
    else:
        # Bad for seller = low price near reservation
        base_price = reservation + buffer  # Close to min willing to accept
        # Add random jitter
        variation = random.uniform(-jitter, jitter)
        naive_price = base_price - variation
        # Clamp to valid range
        naive_price = max(reservation, min(naive_price, pub_max))
        
        # Accept if opponent offers something reasonable
        if state.last_offer_price is not None and state.last_offer_price >= reservation:
            return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(naive_price, 2))


def strategy_micro(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    MiCRO (Minimal Concession Strategy).
    Offers from a pre-sorted grid based on 'step_size'.
    """
    step_size = params.get("step_size", 10.0)
    role = state.role
    my_res = state.true_reservation_price
    
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    # Estimate ZOPA bounds from reservation (standard ZOPA width = 500)
    DEFAULT_ZOPA_WIDTH = 500.0
    if role == "buyer":
        # Buyer's reservation is the high end of ZOPA
        zopa_low = max(pub_min, my_res - DEFAULT_ZOPA_WIDTH)
        zopa_high = my_res
    else:
        # Seller's reservation is the low end of ZOPA
        zopa_low = my_res
        zopa_high = min(pub_max, my_res + DEFAULT_ZOPA_WIDTH)
    
    grid = []
    # Use ceil for low bound to ensure we don't go below ZOPA
    # Use floor for high bound to ensure we don't go above ZOPA
    curr = math.ceil(zopa_low)
    end = math.floor(zopa_high)
    while curr <= end:
        grid.append(float(curr))
        curr += step_size
        
    if role == "buyer":
        sorted_prices = sorted(grid)
    else:
        sorted_prices = sorted(grid, reverse=True)
            
    own_offers = []
    own_offers_set = set()
    opp_offers_set = set()
    last_opp_offer = None
    
    for r, p in state.offer_history:
        if r == role:
            own_offers.append(p)
            own_offers_set.add(p)
        else:
            opp_offers_set.add(p)
            last_opp_offer = p
            
    m = len(own_offers_set)
    n = len(opp_offers_set)
    
    p_new = None
    for p in sorted_prices:
        if p not in own_offers_set:
            p_new = p
            break
            
    def is_acceptable(price):
        if price is None:
            return False
        if role == "buyer":
            return price <= my_res
        else:
            return price >= my_res

    def is_better_or_equal(p1, p2):
        if role == "buyer":
            return p1 <= p2
        else:
            return p1 >= p2

    p_thresh = None
    can_concede = (m <= n)
    
    if can_concede and p_new is not None and is_acceptable(p_new):
        p_thresh = p_new
    else:
        if own_offers:
            if role == "buyer":
                p_thresh = max(own_offers_set)
            else:
                p_thresh = min(own_offers_set)
        else:
            p_thresh = sorted_prices[0] if sorted_prices else my_res

    if last_opp_offer is not None:
        if is_better_or_equal(last_opp_offer, p_thresh) and is_acceptable(last_opp_offer):
            return PriceAction(type="ACCEPT", price=None)
            
    proposal_price = None
    
    if can_concede and p_new is not None and is_acceptable(p_new):
        proposal_price = p_new
    else:
        if own_offers:
            proposal_price = random.choice(own_offers)
        else:
            proposal_price = sorted_prices[0] if sorted_prices else my_res

    return PriceAction(type="OFFER", price=round(proposal_price, 2))


# --- Registry and Metadata ---

@dataclass
class StrategySpec:
    name: str
    description: str
    func: Callable
    default_params: Dict[str, Any]


STRATEGY_REGISTRY: Dict[str, StrategySpec] = {
    # Boulware Spectrum - Selfish start (public bounds, no margin)
    "boulware_very_conceding": StrategySpec(
        "boulware_very_conceding",
        "Rapidly concedes from public bounds (beta=0.2)",
        strategy_boulware,
        {"beta": 0.2}  # No static_margin = start at public bounds (selfish)
    ),
    "boulware_conceding": StrategySpec(
        "boulware_conceding",
        "Concedes moderately from public bounds (beta=0.5)",
        strategy_boulware,
        {"beta": 0.5}  # No static_margin = start at public bounds (selfish)
    ),
    "boulware_firm": StrategySpec(
        "boulware_firm",
        "Concedes slowly from public bounds (beta=2.0)",
        strategy_boulware,
        {"beta": 2.0}  # No static_margin = start at public bounds (selfish)
    ),
    "boulware_hard": StrategySpec(
        "boulware_hard",
        "Concedes very slowly from public bounds (beta=4.0)",
        strategy_boulware,
        {"beta": 4.0}  # No static_margin = start at public bounds (selfish)
    ),
    
    # Noisy Boulware - Same concession curves with bounded randomness
    "noisy_boulware_conceding": StrategySpec(
        "noisy_boulware_conceding",
        "Boulware (beta=0.5) with ±5% ZOPA random noise to keep opponent guessing",
        strategy_noisy_boulware,
        {"beta": 0.5, "noise_max": 25.0, "noise_scale": 0.05}
    ),
    "noisy_boulware_firm": StrategySpec(
        "noisy_boulware_firm",
        "Boulware (beta=2.0) with ±5% ZOPA random noise to keep opponent guessing",
        strategy_noisy_boulware,
        {"beta": 2.0, "noise_max": 25.0, "noise_scale": 0.05}
    ),
    "noisy_boulware_hard": StrategySpec(
        "noisy_boulware_hard",
        "Boulware (beta=4.0) with ±5% ZOPA random noise to keep opponent guessing",
        strategy_noisy_boulware,
        {"beta": 4.0, "noise_max": 25.0, "noise_scale": 0.05}
    ),

    # Tit for Tat - Selfish start (no margin)
    "tit_for_tat": StrategySpec(
        "tit_for_tat",
        "Mirrors opponent concessions from public bounds",
        strategy_tit_for_tat,
        {}  # No initial_margin = start at public bounds
    ),
    
    # Linear / Steady - Selfish start (ZOPA boundary)
    "linear_standard": StrategySpec(
        "linear_standard",
        "Standard linear concession from ZOPA boundary",
        strategy_linear,
        {}  # No static_margin = start at ZOPA boundary
    ),

    # Split Difference - Selfish start (ZOPA boundary)
    "split_difference": StrategySpec(
        "split_difference",
        "Splits difference between last offer and own history",
        strategy_split_difference,
        {}  # No initial_margin = start at ZOPA boundary
    ),
    
    # Hardliner - Selfish start (ZOPA boundary) - stays strictly within ZOPA
    "hardliner": StrategySpec(
        "hardliner",
        "Hold firm at ZOPA boundary until final round",
        strategy_hardliner,
        {}  # No margin = hold at ZOPA boundary
    ),
    
    # Random Oracle
    "random_zopa": StrategySpec(
        "random_zopa",
        "Random offers within ZOPA (Oracle)",
        strategy_random_in_zopa,
        {}
    ),

    # MiCRO Strategies (Minimal Concession)
    "micro_fine": StrategySpec(
        "micro_fine",
        "MiCRO with fine step size (10) - offers from quantized grid",
        strategy_micro,
        {"step_size": 10.0}
    ),
    "micro_moderate": StrategySpec(
        "micro_moderate",
        "MiCRO with moderate step size (25) - offers from quantized grid",
        strategy_micro,
        {"step_size": 25.0}
    ),
    "micro_coarse": StrategySpec(
        "micro_coarse",
        "MiCRO with coarse step size (50) - offers from quantized grid",
        strategy_micro,
        {"step_size": 50.0}
    ),

    # Bad Strategies
    "naive_concession": StrategySpec(
        "naive_concession",
        "Makes consistently bad offers near own reservation (bad strategy)",
        strategy_naive_concession,
        {}
    ),
    "naive_boulware": StrategySpec(
        "naive_boulware",
        "Naive Boulware - concedes in WRONG direction, hardline then sudden drop (bad strategy)",
        strategy_naive_boulware,
        {"beta": 3.0}  # Hardline: flat early, steep late
    )
}


# --- Strategy List Helpers ---

# Strategies excluded from benchmarking and visualization.
# Add any strategy name here to exclude it from all scripts automatically.
EXCLUDED_FROM_BENCHMARK: set = {
    "random_zopa",  # Oracle strategy - requires known ZOPA bounds
}


def get_benchmark_strategies() -> List[str]:
    """Return sorted list of all strategies suitable for benchmarking.
    
    Automatically includes every strategy in STRATEGY_REGISTRY except those
    in EXCLUDED_FROM_BENCHMARK. Add new strategies to the registry and they
    will appear here automatically.
    """
    return sorted(STRATEGY_REGISTRY.keys() - EXCLUDED_FROM_BENCHMARK)


# Strategies that are reactive (depend on opponent offers) vs independent.
# Used by visualize_strategy_curves.py to choose the right plot type.
# Independent strategies are plotted solo; reactive strategies are plotted
# against a standard Boulware opponent.
REACTIVE_STRATEGIES: set = {
    "tit_for_tat",
    "split_difference",
    "micro_fine",
    "micro_moderate",
    "micro_coarse",
    "random_zopa",
}


# --- Deterministic Agent Wrapper ---

class DeterministicPriceAgent(BaseAgent):
    """
    A concrete agent class that uses a strategy function for 'propose_action'.
    No LLM is involved - purely mathematical decisions.
    """
    def __init__(self, agent_id: int, strategy_name: str, strategy_params: Optional[Dict] = None):
        super().__init__(agent_id, "deterministic", "none")
        
        self.strategy_name = strategy_name
        
        spec = STRATEGY_REGISTRY.get(strategy_name)
        if not spec:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        self.strategy_func = spec.func
        self.params = spec.default_params.copy()
        if strategy_params:
            self.params.update(strategy_params)
            
        self.description = spec.description
        
    async def generate_response(self) -> str:
        raise NotImplementedError("DeterministicPriceAgent is for dataset_mode only (no LLM text).")
        
    def add_to_memory(self, role: str, content: str):
        pass
        
    def reset_memory(self):
        pass
        
    def propose_action(self, state: PriceState) -> PriceAction:
        """Delegates to the strategy function."""
        return self.strategy_func(state, self.params)

    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        return True

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_type": "deterministic_price_agent",
            "strategy": self.strategy_name,
            "strategy_params": self.params,
            "strategy_description": self.description
        }
