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


def strategy_price_fixed(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Fixed Price strategy: Always offers reservation +/- margin.
    """
    margin = params.get("margin", 0.0)
    reservation = state.effective_reservation_price
    
    if state.role == "buyer":
        target_price = reservation - margin
        if state.last_offer_price is not None and state.last_offer_price <= target_price:
            return PriceAction(type="ACCEPT", price=None)
    else:
        target_price = reservation + margin
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


def strategy_naive_concession(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Naive Concession strategy - starts with one normal offer, then immediately concedes to a terrible position.
    This is a "bad" strategy that demonstrates poor negotiation behavior.
    First turn: offers at ZOPA boundary (normal selfish start)
    Subsequent turns: offers very close to opponent's reservation (giving away the surplus).
    """
    reservation = state.effective_reservation_price
    role = state.role
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)

    # Count own offers to see if this is the first turn
    own_offers = [p for r, p in state.offer_history if r == role]
    is_first_offer = len(own_offers) == 0

    # Calculate symmetric starting positions around ZOPA
    DEFAULT_ZOPA_WIDTH = 500.0
    
    if is_first_offer:
        # First offer: start symmetrically around ZOPA
        if role == "buyer":
            first_price = max(pub_min, reservation - DEFAULT_ZOPA_WIDTH)  # Start 500 below reservation
        else:
            first_price = min(pub_max, reservation + DEFAULT_ZOPA_WIDTH)  # Start 500 above reservation
        return PriceAction(type="OFFER", price=round(first_price, 2))
    
    # Subsequent offers: immediately concede to a terrible position (near own reservation)
    # This is a "bad" strategy that gives away almost all surplus
    buffer_pct = 0.05  # Only keep 5% margin from own reservation
    
    if role == "buyer":
        # Buyer: offers very close to their own max (terrible deal for buyer)
        buffer = (reservation - pub_min) * buffer_pct
        naive_price = min(reservation, reservation - buffer)

        if state.last_offer_price is not None and state.last_offer_price <= naive_price:
            return PriceAction(type="ACCEPT", price=None)
    else:
        # Seller: offers very close to their own min (terrible deal for seller)
        buffer = (pub_max - reservation) * buffer_pct
        naive_price = max(reservation, reservation + buffer)

        if state.last_offer_price is not None and state.last_offer_price >= naive_price:
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
    
    # Use ZOPA bounds if provided, otherwise use public bounds
    zopa_low = params.get("zopa_low", pub_min)
    zopa_high = params.get("zopa_high", pub_max)
    
    grid = []
    curr = math.floor(zopa_low)
    end = math.ceil(zopa_high)
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
    
    # Boulware Spectrum - With margin (for comparison, starts closer to reservation)
    "boulware_very_conceding_margin": StrategySpec(
        "boulware_very_conceding_margin",
        "Rapidly concedes from margin (beta=0.2, margin=400)",
        strategy_boulware,
        {"beta": 0.2, "static_margin": 400.0}
    ),
    "boulware_conceding_margin": StrategySpec(
        "boulware_conceding_margin",
        "Concedes moderately from margin (beta=0.5, margin=400)",
        strategy_boulware,
        {"beta": 0.5, "static_margin": 400.0}
    ),
    "boulware_firm_margin": StrategySpec(
        "boulware_firm_margin",
        "Concedes slowly from margin (beta=2.0, margin=400)",
        strategy_boulware,
        {"beta": 2.0, "static_margin": 400.0}
    ),
    "boulware_hard_margin": StrategySpec(
        "boulware_hard_margin",
        "Concedes very slowly from margin (beta=4.0, margin=400)",
        strategy_boulware,
        {"beta": 4.0, "static_margin": 400.0}
    ),
    
    # Price Fixed
    "price_fixed_strict": StrategySpec(
        "price_fixed_strict",
        "Offers exactly reservation +/- small margin",
        strategy_price_fixed,
        {"margin": 20.0}
    ),
    "price_fixed_loose": StrategySpec(
        "price_fixed_loose",
        "Offers reservation +/- large margin",
        strategy_price_fixed,
        {"margin": 100.0}
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
    
    # Time Dependent - Selfish start (ZOPA boundary)
    "time_dependent": StrategySpec(
        "time_dependent",
        "Acceptance threshold relaxes over time from ZOPA boundary",
        strategy_time_dependent_threshold,
        {}  # No margin = start at ZOPA boundary
    ),
    
    # Hardliner - Selfish start (ZOPA boundary) - stays strictly within ZOPA
    "hardliner": StrategySpec(
        "hardliner",
        "Hold firm at ZOPA boundary until final round",
        strategy_hardliner,
        {}  # No margin = hold at ZOPA boundary
    ),
    
    # Hardliner with margin (for comparison)
    "hardliner_margin": StrategySpec(
        "hardliner_margin",
        "Hold firm with margin until final round",
        strategy_hardliner,
        {"margin": 400.0}
    ),
    
    # Random Oracle
    "random_zopa": StrategySpec(
        "random_zopa",
        "Random offers within ZOPA (Oracle)",
        strategy_random_in_zopa,
        {}
    ),
    
    # MiCRO Strategies
    "micro_fine": StrategySpec(
        "micro_fine",
        "MiCRO agent with fine grid (step=5.0)",
        strategy_micro,
        {"step_size": 5.0}
    ),
    "micro_moderate": StrategySpec(
        "micro_moderate",
        "MiCRO agent with moderate grid (step=25.0)",
        strategy_micro,
        {"step_size": 25.0}
    ),
    "micro_coarse": StrategySpec(
        "micro_coarse",
        "MiCRO agent with coarse grid (step=100.0)",
        strategy_micro,
        {"step_size": 100.0}
    ),

    # Bad Strategy - Naive Concession (concedes immediately to terrible position)
    "naive_concession": StrategySpec(
        "naive_concession",
        "Immediately concedes 80% of ZOPA and sticks with it (bad strategy)",
        strategy_naive_concession,
        {}
    )
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
