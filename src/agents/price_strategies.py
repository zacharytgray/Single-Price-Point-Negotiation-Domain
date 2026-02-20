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
    Split difference between last offer and own previous offer (or ZOPA boundary).
    Starts at the opponent's estimated reservation price (ZOPA boundary), then on each
    subsequent turn proposes the midpoint between own last offer and the opponent's last offer.

    The ZOPA boundary is estimated as reservation +/- DEFAULT_ZOPA_WIDTH when not explicitly
    provided via params (zopa_low / zopa_high / initial_margin).
    """
    role = state.role
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)

    # Estimate the selfish starting anchor: opponent's reservation = own reservation +/- ZOPA width.
    # This mirrors how every other strategy initialises (e.g. boulware, hardliner).
    DEFAULT_ZOPA_WIDTH = 500.0
    margin = params.get("initial_margin", DEFAULT_ZOPA_WIDTH)
    if role == "buyer":
        # Buyer starts LOW: own reservation minus ZOPA width (≈ seller's reservation)
        selfish_start = max(pub_min, reservation - margin)
    else:
        # Seller starts HIGH: own reservation plus ZOPA width (≈ buyer's reservation)
        selfish_start = min(pub_max, reservation + margin)

    # --- First offer (no prior offers at all, including when seller goes second) ---
    # Use selfish_start as the anchor so the midpoint mechanism begins from the right place.
    my_offers = [p for r, p in state.offer_history if r == role]

    if not my_offers:
        # No offer from us yet. If the opponent has already made an offer, split the
        # difference between our selfish start and their offer; otherwise just open
        # at the selfish start.
        if state.last_offer_price is not None:
            opp_price = state.last_offer_price
            midpoint = (selfish_start + opp_price) / 2.0
            if role == "buyer":
                target = min(midpoint, reservation)
                if opp_price <= target:
                    return PriceAction(type="ACCEPT", price=None)
            else:
                target = max(midpoint, reservation)
                if opp_price >= target:
                    return PriceAction(type="ACCEPT", price=None)
            return PriceAction(type="OFFER", price=round(target, 2))
        else:
            return PriceAction(type="OFFER", price=round(selfish_start, 2))

    # --- Subsequent offers: split between own last offer and opponent's last offer ---
    opp_price = state.last_offer_price  # always set here (we've already offered at least once)
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


def strategy_fair(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Fair / Equitable strategy.

    Targets the midpoint of the ZOPA (equal utility split for both parties) and
    converges toward it linearly over the negotiation. On each turn it proposes
    the price that would give both buyer and seller exactly equal shares of the
    surplus, then relaxes toward that target as the deadline approaches.

    Acceptance rule: accept any offer that is at least as good as the current
    fair-target (i.e. the opponent is already offering a fair or better deal).

    Parameters:
        convergence_rate (float): How quickly to move toward the midpoint.
            0.0 = open at ZOPA boundary and never move (like hardliner).
            1.0 = open directly at the midpoint on turn 1 (default: 0.8).
        noise (float): Optional small random jitter around the fair price
            to avoid being perfectly predictable (default: 0.0).
    """
    role = state.role
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)

    DEFAULT_ZOPA_WIDTH = 500.0

    # Estimate the ZOPA midpoint (equal-utility price) from own reservation.
    # Buyer's reservation = buyer_max  -> midpoint = buyer_max - ZOPA_WIDTH/2
    # Seller's reservation = seller_min -> midpoint = seller_min + ZOPA_WIDTH/2
    # Both expressions yield the same price when ZOPA_WIDTH is fixed.
    zopa_width = params.get("zopa_width", DEFAULT_ZOPA_WIDTH)
    fair_price = reservation + (zopa_width / 2.0) if role == "seller" else reservation - (zopa_width / 2.0)
    fair_price = max(pub_min, min(pub_max, fair_price))

    # Selfish starting anchor: ZOPA boundary (opponent's reservation estimate)
    if role == "buyer":
        selfish_start = max(pub_min, reservation - zopa_width)
    else:
        selfish_start = min(pub_max, reservation + zopa_width)

    # Convergence: interpolate from selfish_start toward fair_price over time.
    # convergence_rate=1.0 means we open at fair_price immediately.
    convergence_rate = params.get("convergence_rate", 0.8)
    time_frac = min(1.0, state.timestep / state.max_turns)
    blend = min(1.0, convergence_rate + time_frac * (1.0 - convergence_rate))
    target = selfish_start + blend * (fair_price - selfish_start)

    # Clamp to own reservation (never offer beyond own limit)
    if role == "buyer":
        target = min(target, reservation)
    else:
        target = max(target, reservation)

    # Optional jitter
    noise_amt = params.get("noise", 0.0)
    if noise_amt > 0.0:
        target += random.uniform(-noise_amt, noise_amt)
        target = max(pub_min, min(pub_max, target))
        if role == "buyer":
            target = min(target, reservation)
        else:
            target = max(target, reservation)

    # Accept if the opponent's offer is at least as fair as the ZOPA midpoint.
    # We accept anything at or better than fair_price regardless of where we are
    # in our convergence curve — there's no reason to reject an already-fair deal.
    # We also accept anything better than our current (possibly more selfish) target.
    if state.last_offer_price is not None:
        if role == "buyer" and (state.last_offer_price <= fair_price or state.last_offer_price <= target):
            return PriceAction(type="ACCEPT", price=None)
        if role == "seller" and (state.last_offer_price >= fair_price or state.last_offer_price >= target):
            return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=round(target, 2))


# ---------------------------------------------------------------------------
# ChargingBoul — Adaptive opponent-modelling strategy
# ---------------------------------------------------------------------------

# Persistent cross-round memory keyed by opponent_id.
# Each entry: {"opp_offers": List[float], "ubi": int, "aui": int,
#              "classification": str, "best_received_utility": float}
charging_boul_memory: Dict[str, Dict] = {}


def calculate_ubi(received_prices: List[float]) -> int:
    """
    Unique Bid Index (UBI).

    Recursively splits the opponent's offer list in half.  If the right half
    contains *more* unique prices than the left half the opponent is still
    exploring new territory late in the negotiation — increment UBI and recurse
    on the right half.  Stop as soon as the right half is no longer more
    diverse.

    A high UBI (≥ 5) indicates a Boulwarish opponent who delays real concessions
    until late in the game.
    """
    def _recurse(prices: List[float]) -> int:
        if len(prices) < 2:
            return 0
        mid = len(prices) // 2
        left, right = prices[:mid], prices[mid:]
        if len(set(right)) > len(set(left)):
            return 1 + _recurse(right)
        return 0

    return _recurse(received_prices)


def calculate_aui(received_prices: List[float], state: PriceState) -> int:
    """
    Average Utility Index (AUI).

    Converts each opponent offer into *our* utility, then recursively checks
    whether the right half of the utility sequence has a higher mean than the
    left half.  Each time it does, AUI is incremented and we recurse on the
    right half.

    A high AUI means the opponent is genuinely conceding (improving our
    utility) over time — i.e. a Conceder.  A low AUI (≤ 2) means they are
    not conceding — i.e. a Hardliner.
    """
    reservation = state.effective_reservation_price
    role = state.role

    def price_to_our_utility(price: float) -> float:
        if role == "buyer":
            return reservation - price   # lower price → higher utility for buyer
        else:
            return price - reservation   # higher price → higher utility for seller

    utilities = [price_to_our_utility(p) for p in received_prices]

    def _recurse(utils: List[float]) -> int:
        if len(utils) < 2:
            return 0
        mid = len(utils) // 2
        left, right = utils[:mid], utils[mid:]
        if (sum(right) / len(right)) > (sum(left) / len(left)):
            return 1 + _recurse(right)
        return 0

    return _recurse(utilities)


def _cb_classify(ubi: int, aui: int) -> str:
    """Return 'boulwarish', 'hardliner', or 'conceder'."""
    if ubi >= 5:
        return "boulwarish"
    if aui <= 2:
        return "hardliner"
    return "conceder"


def strategy_charging_boul(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    ChargingBoul — Faithful scalar-domain adaptation of Shymanski (2025).

    Reference: "ChargingBoul: A Competitive Negotiating Agent with Novel
    Opponent Modeling", arXiv:2512.06595.

    The paper operates in normalised utility space [0, 1].  In our scalar
    price domain, utility maps directly to price via the reservation price:

        buyer  utility = (reservation − price) / ZOPA_WIDTH   ∈ [0, 1]
        seller utility = (price − reservation) / ZOPA_WIDTH   ∈ [0, 1]

    So a utility target g_t ∈ [0,1] converts to a price target as:

        buyer  target_price = reservation − g_t * ZOPA_WIDTH
        seller target_price = reservation + g_t * ZOPA_WIDTH

    This keeps all bids inside the ZOPA (between reservation prices).

    ── Concession curve (paper Eq. 1) ──────────────────────────────────
        g(t) = m + (1 − m)(1 − t^(1/E))

    where:
        t = timestep / max_turns  ∈ (0, 1]
        m = minimum acceptable utility (default 0.5)
        E = concession rate (default 0.1; smaller = later concessions)

    At t=0: g=1 (maximum utility — most selfish bid)
    At t=1: g=m (minimum utility floor)

    Note: with E=0.1 the exponent 1/E=10 makes the curve very flat early
    and steep late — this is intentional Boulware behaviour per the paper.

    ── Opponent adaptation (paper Eq. 3) ────────────────────────────────
    Against a Boulwarish opponent (UBI ≥ 5):
        E = 0.2 × 2^(5 − ubi)
    This mirrors the opponent's lateness so neither side is exploited.

    Against a Conceder: m lowered to 0.4 (paper §3.2.1).
    Against a Hardliner: no change (paper §3.2.2).

    ── Bid window (paper Eq. 2) ─────────────────────────────────────────
        [g(t) − (3t+1)ε,  g(t) + (3t+1)ε]   in utility space
    Converted to price space and clamped to [reservation, ZOPA boundary].

    ── Late-round logic (paper §3.2.2, Eq. 4) ───────────────────────────
    Late period: t > 1 − 0.5^ubi
    Against Boulwarish: lower m to 0.3; re-propose best received bid if
    its utility > m and predicted opponent utility < 2m.

    ── Acceptance (paper §3.3) ──────────────────────────────────────────
    Generate a candidate bid first.  Accept iff received offer is better
    than the generated bid (and within reservation).

    Parameters (all optional, passed via params dict):
        opponent_id  (str)   : key for persistent memory (default "default").
        m_default    (float) : minimum utility floor (default 0.5, per paper).
        E_default    (float) : concession rate — smaller = more Boulwarish.
                               Paper used 0.1 for 50-round games; use 0.4 for
                               20-turn games to get an equivalent visible curve.
        epsilon      (float) : bid-window tolerance as utility fraction (default 0.02).
        zopa_width   (float) : ZOPA width for utility↔price conversion (default 500).
    """
    role        = state.role
    reservation = state.effective_reservation_price
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)

    # ZOPA width used for utility ↔ price conversion.
    # Utility 1.0 = capturing the full ZOPA; utility 0.0 = at own reservation.
    ZOPA_WIDTH: float = params.get("zopa_width", 500.0)

    # ------------------------------------------------------------------ #
    # 1. Update persistent opponent memory                                #
    # ------------------------------------------------------------------ #
    opponent_id: str = params.get("opponent_id", "default")

    if opponent_id not in charging_boul_memory:
        charging_boul_memory[opponent_id] = {
            "opp_offers": [],
            "ubi": 0,
            "aui": 0,
            "classification": "conceder",   # optimistic default until we have data
            "best_received_utility": -float("inf"),
        }

    mem = charging_boul_memory[opponent_id]

    # Collect opponent offers from this episode's history
    opp_role = "seller" if role == "buyer" else "buyer"
    mem["opp_offers"] = [p for r, p in state.offer_history if r == opp_role]

    # Recompute indices once we have at least 4 opponent offers
    if len(mem["opp_offers"]) >= 4:
        mem["ubi"] = calculate_ubi(mem["opp_offers"])
        mem["aui"] = calculate_aui(mem["opp_offers"], state)
        mem["classification"] = _cb_classify(mem["ubi"], mem["aui"])

    # Track the best (most favourable) utility received from the opponent
    if state.last_offer_price is not None:
        received_util = (
            (reservation - state.last_offer_price) / ZOPA_WIDTH if role == "buyer"
            else (state.last_offer_price - reservation) / ZOPA_WIDTH
        )
        if received_util > mem["best_received_utility"]:
            mem["best_received_utility"] = received_util
            mem["best_received_price"]   = state.last_offer_price

    # ------------------------------------------------------------------ #
    # 2. Concession curve parameters (paper §3.2.1)                       #
    # ------------------------------------------------------------------ #
    classification = mem["classification"]
    ubi            = mem["ubi"]

    m_default: float = params.get("m_default", 0.5)   # paper default: 0.5
    E_default: float = params.get("E_default", 0.1)   # paper default: 0.1 (calibrated for 50 rounds)

    m = m_default
    E = E_default

    # E controls the concession curve shape via exponent 1/E:
    #   Small E (e.g. 0.1) → exponent 10 → very flat early, steep at deadline (Boulwarish).
    #   Large E (e.g. 1.0) → exponent 1  → linear concession.
    # The paper used E=0.1 for 50-round competitions; for 20-turn games E=0.4
    # produces an equivalent Boulwarish shape (slow early, visible mid-game movement).
    E = E_default

    if classification == "conceder":
        m = 0.4   # paper §3.2.1: lower m against conceder to ensure agreement
        E = min(E * 2.0, 1.0)   # concede faster against a conceder
    elif classification == "boulwarish":
        # Paper Eq. 3: mirror the opponent's lateness with a smaller E.
        # Smaller E → larger exponent → curve stays high longer.
        E_boul = 0.2 * (2 ** (5 - ubi))
        E = max(0.01, min(E_boul, E))   # never concede faster than default
    # hardliner: keep defaults (paper §3.2.2 — no special concession)

    # ------------------------------------------------------------------ #
    # 3. Time fraction and concession curve g(t) (paper Eq. 1)            #
    # ------------------------------------------------------------------ #
    t = state.timestep / state.max_turns   # ∈ (0, 1]
    t = min(t, 0.9999)                     # avoid t=1 numerical edge

    # g(t) = m + (1 − m)(1 − t^(1/E))
    # Exponent 1/E: larger = more Boulwarish (flatter early, steeper late).
    # E=0.4 → exponent 2.5: visibly Boulwarish over 20 turns.
    # E=0.1 → exponent 10:  nearly flat for 18/20 turns (paper's 50-round calibration).
    g_t = m + (1.0 - m) * (1.0 - t ** (1.0 / E))
    g_t = max(m, min(1.0, g_t))

    # ------------------------------------------------------------------ #
    # 4. Late-round concession logic (paper §3.2.2, Eq. 4)                #
    # ------------------------------------------------------------------ #
    # Paper Eq. 4: late period when t > 1 − 0.5/max(ubi, 1)
    # (The paper writes "1 - .5^ubi" but the intent from Fig. 3 is that
    # ubi=5 triggers the late period at t≈0.5, matching 0.5/5=0.1 from end.)
    late_threshold = 1.0 - 0.5 / max(ubi, 1)
    in_late_period = t > late_threshold

    if in_late_period and classification == "boulwarish":
        m_late = 0.3   # paper: lower m to 0.3 in late period
        g_t    = max(m_late, g_t)
        best_util = mem.get("best_received_utility", -float("inf"))
        # Re-propose best received bid if its utility > m_late (paper §3.2.2)
        if best_util > m_late and "best_received_price" in mem:
            candidate_price = mem["best_received_price"]
            if role == "buyer" and candidate_price <= reservation:
                if state.last_offer_price is not None and state.last_offer_price <= candidate_price:
                    return PriceAction(type="ACCEPT", price=None)
                return PriceAction(type="OFFER", price=round(candidate_price, 2))
            elif role == "seller" and candidate_price >= reservation:
                if state.last_offer_price is not None and state.last_offer_price >= candidate_price:
                    return PriceAction(type="ACCEPT", price=None)
                return PriceAction(type="OFFER", price=round(candidate_price, 2))

    # ------------------------------------------------------------------ #
    # 5. Convert utility target g(t) to a price                           #
    # ------------------------------------------------------------------ #
    # The paper's utility function: u(price) ∈ [0,1] where
    #   u=1 → own reservation (best possible outcome for self)
    #   u=0 → opponent's reservation / ZOPA boundary (worst acceptable)
    #
    # g(t) starts at 1 (most selfish) and declines to m (minimum floor).
    # So g(t)=1 → offer at own reservation; g(t)=0 → offer at ZOPA boundary.
    #
    # In our scalar domain, inverting the utility function:
    #   buyer  utility = (reservation − price) / ZOPA_WIDTH
    #   → price = reservation − u * ZOPA_WIDTH
    #   g(t)=1 → price = reservation − ZOPA_WIDTH  (= ZOPA boundary, most selfish opening)
    #   g(t)=m → price = reservation − m * ZOPA_WIDTH  (= minimum concession floor)
    #
    # Wait — that maps g=1 to the ZOPA boundary (lowest price for buyer),
    # which IS the most selfish opening bid (buyer wants low prices).
    # And g=m maps to reservation − m*ZOPA_WIDTH, which is closer to reservation.
    # So the concession goes from ZOPA_boundary → toward reservation as g decreases.
    # This is correct: buyer opens at ZOPA boundary and concedes toward reservation.
    #
    # The issue was that with m=0.5, the buyer only ever reaches the midpoint
    # (reservation − 0.5*ZOPA = midpoint), never going above it. That's intentional:
    # m is the minimum utility the agent will accept.
    #
    # The flatness problem: g(t) barely changes from 1.0 for the first 80% of turns.
    # Solution: use (1 − g(t)) as the concession fraction so that:
    #   t=0 → concession=0 → price at ZOPA boundary (most selfish)
    #   t=1 → concession=(1−m) → price at reservation − m*ZOPA (minimum floor)
    #
    # concession_frac = 1 − g(t)  ∈ [0, 1−m]
    # buyer  target = zopa_boundary + concession_frac * ZOPA_WIDTH
    #               = (reservation − ZOPA_WIDTH) + (1−g_t) * ZOPA_WIDTH
    # seller target = zopa_boundary − concession_frac * ZOPA_WIDTH
    #               = (reservation + ZOPA_WIDTH) − (1−g_t) * ZOPA_WIDTH
    concession_frac = 1.0 - g_t   # ∈ [0, 1−m]; 0=most selfish, grows as g_t falls

    zopa_low_est  = max(pub_min, reservation - ZOPA_WIDTH)   # ≈ seller_min
    zopa_high_est = min(pub_max, reservation + ZOPA_WIDTH)   # ≈ buyer_max

    if role == "buyer":
        # Opens at zopa_low_est (most selfish), concedes toward reservation
        target_price = zopa_low_est + concession_frac * ZOPA_WIDTH
        target_price = max(zopa_low_est, min(target_price, reservation))
    else:
        # Opens at zopa_high_est (most selfish), concedes toward reservation
        target_price = zopa_high_est - concession_frac * ZOPA_WIDTH
        target_price = max(reservation, min(target_price, zopa_high_est))

    # ------------------------------------------------------------------ #
    # 6. Bid randomisation window (paper Eq. 2)                           #
    # ------------------------------------------------------------------ #
    # Paper: window = [g(t)−(3t+1)ε, g(t)+(3t+1)ε] in utility space.
    # ε is the tolerance (granularity of bid space), typically 0.001–0.05.
    # We convert the utility window to a price window.
    epsilon: float = params.get("epsilon", 0.02)   # utility-space tolerance
    half_window = (3.0 * t + 1.0) * epsilon * ZOPA_WIDTH   # price-space half-width

    if role == "buyer":
        # Buyer wants LOW prices; window stays within [zopa_low_est, reservation]
        lo = max(zopa_low_est, target_price - half_window)
        hi = min(reservation,  target_price + half_window)
    else:
        # Seller wants HIGH prices; window stays within [reservation, zopa_high_est]
        lo = max(reservation,   target_price - half_window)
        hi = min(zopa_high_est, target_price + half_window)

    if lo > hi:
        lo, hi = hi, lo
    bid_price = lo if lo == hi else random.uniform(lo, hi)
    bid_price = round(bid_price, 2)

    # Paper §3.2.1: "If ChargingBoul selects a bid with lower utility than one
    # it has already *received and rejected*, it re-proposes that rejected bid."
    # i.e. if our new bid is worse for us than the best offer we've received,
    # re-propose that received offer instead (never undersell ourselves).
    best_recv_util = mem.get("best_received_utility", -float("inf"))
    if best_recv_util > -float("inf") and "best_received_price" in mem:
        recv_price = mem["best_received_price"]
        # Check if the received price is better for us than our current bid
        if role == "buyer" and recv_price < bid_price:
            # Received offer is lower (better for buyer) than what we'd propose
            bid_price = recv_price
        elif role == "seller" and recv_price > bid_price:
            # Received offer is higher (better for seller) than what we'd propose
            bid_price = recv_price

    # Final safety clamp — ensure bid never crosses own reservation
    if role == "buyer":
        bid_price = max(zopa_low_est, min(bid_price, reservation))
    else:
        bid_price = max(reservation, min(bid_price, zopa_high_est))

    # ------------------------------------------------------------------ #
    # 7. Acceptance rule                                                   #
    # ------------------------------------------------------------------ #
    if state.last_offer_price is not None:
        opp_price = state.last_offer_price
        # Check the offer is within our reservation (acceptable)
        is_acceptable = (
            opp_price <= reservation if role == "buyer" else opp_price >= reservation
        )
        # Check it is at least as good as our generated bid
        is_good_enough = (
            opp_price <= bid_price if role == "buyer" else opp_price >= bid_price
        )
        if is_acceptable and is_good_enough:
            return PriceAction(type="ACCEPT", price=None)

    return PriceAction(type="OFFER", price=bid_price)


# ---------------------------------------------------------------------------


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

    # ChargingBoul — Adaptive opponent-modelling (Shymanski 2025, arXiv:2512.06595)
    "charging_boul": StrategySpec(
        "charging_boul",
        "ChargingBoul: adaptive Boulware with UBI/AUI opponent modelling (m=0.5, E=0.4)",
        strategy_charging_boul,
        {"m_default": 0.5, "E_default": 0.4, "epsilon": 0.02}
    ),
    "charging_boul_aggressive": StrategySpec(
        "charging_boul_aggressive",
        "ChargingBoul with higher utility floor (m=0.6, E=0.2) — strong Boulware, less willing to concede",
        strategy_charging_boul,
        {"m_default": 0.6, "E_default": 0.2, "epsilon": 0.02}
    ),
    "charging_boul_patient": StrategySpec(
        "charging_boul_patient",
        "ChargingBoul with faster concession rate (E=0.8) — more linear, concedes steadily",
        strategy_charging_boul,
        {"m_default": 0.5, "E_default": 0.8, "epsilon": 0.02}
    ),

    # Fair / Equitable
    "fair": StrategySpec(
        "fair",
        "Targets equal ZOPA split (midpoint), converges toward fair price over time",
        strategy_fair,
        {"convergence_rate": 0.8}
    ),
    "fair_fast": StrategySpec(
        "fair_fast",
        "Opens directly at ZOPA midpoint and holds (convergence_rate=1.0)",
        strategy_fair,
        {"convergence_rate": 1.0}
    ),
    "fair_slow": StrategySpec(
        "fair_slow",
        "Starts at ZOPA boundary and slowly converges to midpoint (convergence_rate=0.4)",
        strategy_fair,
        {"convergence_rate": 0.4}
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
    "charging_boul",
    "charging_boul_aggressive",
    "charging_boul_patient",
    "fair",
    "fair_fast",
    "fair_slow",
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
