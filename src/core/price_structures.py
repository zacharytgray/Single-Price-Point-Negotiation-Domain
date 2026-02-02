from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple, Dict, Any

@dataclass
class PriceAction:
    """Canonical action structure for price domain."""
    type: Literal["OFFER", "ACCEPT"]
    price: Optional[float] = None
    
    def __post_init__(self):
        if self.type == "OFFER" and self.price is None:
            raise ValueError("OFFER action requires a price.")
        if self.type == "OFFER" and self.price is not None:
            # Ensure price is a float
            self.price = float(self.price)
        if self.type == "ACCEPT" and self.price is not None:
             raise ValueError("ACCEPT action must have price=None.")

@dataclass
class PriceState:
    """Canonical state structure for price domain."""
    timestep: int
    max_turns: int
    role: Literal["buyer", "seller"]
    last_offer_price: Optional[float]
    offer_history: List[Tuple[str, float]]  # list of (proposer_role, price)
    effective_reservation_price: float  # Point of indifference
    true_reservation_price: float  # Raw private value (Buyer Max or Seller Min)
    public_price_range: Optional[Tuple[float, float]] = None  # (min, max) if known publicly

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "max_turns": self.max_turns,
            "role": self.role,
            "last_offer_price": self.last_offer_price,
            "offer_history": self.offer_history,
            "effective_reservation_price": self.effective_reservation_price,
            "true_reservation_price": self.true_reservation_price,
            "public_price_range": self.public_price_range
        }
