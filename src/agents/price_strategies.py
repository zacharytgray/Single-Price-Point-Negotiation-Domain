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
    """
    beta = params.get("beta", 1.0)
    reservation = state.effective_reservation_price

    # Cap time_frac to prevent full capitulation
    concession_cap = params.get("concession_cap", 0.95)
    time_frac = min(concession_cap, state.timestep / state.max_turns)
    
    margin = params.get("static_margin", 50.0)
    
    if state.role == "buyer":
        start_price = max(0.0, reservation - margin)
        target_price = start_price + (reservation - start_price) * (time_frac ** beta)
        target_price = min(target_price, reservation)
        
    else:  # seller
        start_price = reservation + margin
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
    Start with an extreme offer.
    """
    role = state.role
    reservation = state.effective_reservation_price
    initial_margin = params.get("initial_margin", 50.0)
    
    # Initial offer or if no history
    if not state.offer_history:
        if role == "buyer":
            return PriceAction(type="OFFER", price=round(reservation - initial_margin, 2))
        else:
            return PriceAction(type="OFFER", price=round(reservation + initial_margin, 2))
            
    # Filter history
    my_offers = [p for r, p in state.offer_history if r == role]
    opp_offers = [p for r, p in state.offer_history if r != role]
    
    if not my_offers:
        if role == "buyer":
            target = reservation - initial_margin
        else:
            target = reservation + initial_margin
         
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
    """
    role = state.role
    reservation = state.effective_reservation_price
    
    if state.last_offer_price is None:
        margin = params.get("initial_margin", 50.0)
        target = reservation - margin if role == "buyer" else reservation + margin
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
    """
    reservation = state.effective_reservation_price
    role = state.role
    margin = params.get("margin", 20.0)
    
    t = state.timestep
    T = state.max_turns
    
    concession_cap = params.get("concession_cap", 0.95)
    frac = min(concession_cap, t / T)
    
    if role == "buyer":
        current_threshold = (reservation - margin) + (margin * frac)
        target = min(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price <= target:
            return PriceAction(type="ACCEPT", price=None)
    else:
        current_threshold = (reservation + margin) - (margin * frac)
        target = max(current_threshold, reservation)
        
        if state.last_offer_price is not None and state.last_offer_price >= target:
            return PriceAction(type="ACCEPT", price=None)
            
    return PriceAction(type="OFFER", price=round(target, 2))


def strategy_hardliner(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    Maintains a tough stance until the very last round, then concedes to reservation.
    """
    reservation = state.effective_reservation_price
    role = state.role
    margin = params.get("margin", 30.0)
    
    if state.timestep >= state.max_turns - 1:
        cave_in_margin = params.get("cave_in_margin", 5.0)
        if role == "buyer":
            target = reservation - cave_in_margin
        else:
            target = reservation + cave_in_margin
    else:
        if role == "buyer":
            target = reservation - margin
        else:
            target = reservation + margin
            
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


def strategy_micro(state: PriceState, params: Dict[str, Any]) -> PriceAction:
    """
    MiCRO (Minimal Concession Strategy).
    Offers from a pre-sorted grid based on 'step_size'.
    """
    step_size = params.get("step_size", 10.0)
    role = state.role
    my_res = state.true_reservation_price
    
    pub_min, pub_max = state.public_price_range if state.public_price_range else (0.0, 2000.0)
    
    grid = []
    curr = math.floor(pub_min)
    end = math.ceil(pub_max)
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
    # Boulware Spectrum
    "boulware_very_conceding": StrategySpec(
        "boulware_very_conceding",
        "Rapidly concedes early (beta=0.2)",
        strategy_boulware,
        {"beta": 0.2, "static_margin": 400.0}
    ),
    "boulware_conceding": StrategySpec(
        "boulware_conceding",
        "Concedes moderately early (beta=0.5)",
        strategy_boulware,
        {"beta": 0.5, "static_margin": 400.0}
    ),
    "boulware_linear": StrategySpec(
        "boulware_linear",
        "Linear concession (beta=1.0)",
        strategy_boulware,
        {"beta": 1.0, "static_margin": 400.0}
    ),
    "boulware_firm": StrategySpec(
        "boulware_firm",
        "Concedes slowly (beta=2.0)",
        strategy_boulware,
        {"beta": 2.0, "static_margin": 400.0}
    ),
    "boulware_hard": StrategySpec(
        "boulware_hard",
        "Concedes very slowly (beta=4.0)",
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

    # Tit for Tat
    "tit_for_tat": StrategySpec(
        "tit_for_tat",
        "Mirrors opponent concessions",
        strategy_tit_for_tat,
        {"initial_margin": 100.0}
    ),
    
    # Linear / Steady
    "linear_standard": StrategySpec(
        "linear_standard",
        "Standard linear concession",
        strategy_linear,
        {"static_margin": 400.0}
    ),

    # Split Difference
    "split_difference": StrategySpec(
        "split_difference",
        "Splits difference between last offer and own history",
        strategy_split_difference,
        {"initial_margin": 400.0}
    ),
    
    # Time Dependent
    "time_dependent": StrategySpec(
        "time_dependent",
        "Acceptance threshold relaxes over time",
        strategy_time_dependent_threshold,
        {"margin": 200.0}
    ),
    
    # Hardliner
    "hardliner": StrategySpec(
        "hardliner",
        "Hold firm until final round",
        strategy_hardliner,
        {"margin": 400.0}
    ),
    
    # True Hardliner (No Cave-in)
    "true_hardliner": StrategySpec(
        "true_hardliner",
        "Never concedes, never caves in",
        strategy_hardliner,
        {"margin": 400.0, "cave_in_margin": 400.0}
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
