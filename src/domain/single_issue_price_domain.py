"""
Single Issue Price Domain implementation.
Handles buyer-seller price negotiation logic.
"""

import re
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal


@dataclass
class ParsedAction:
    """Parsed action from agent text response."""
    action_type: Literal["OFFER", "ACCEPT", "INVALID"]
    offer_content: Optional[float]  # Price for OFFER, None for ACCEPT
    raw_text: str


class SingleIssuePriceDomain:
    """
    Domain for single-issue price negotiation.
    Seller wants higher price, Buyer wants lower price.
    """
    
    def __init__(self):
        self.domain_name = "single_issue_price"
        self.buyer_id = 1  # Convention: Agent 1 is Buyer
        self.seller_id = 2  # Convention: Agent 2 is Seller
        
        # State
        self.round_id: int = 0
        self.current_offer: Optional[float] = None
        self.buyer_max: float = 0.0
        self.seller_min: float = 0.0
        self.agreement_price: Optional[float] = None
        self.last_offerer_id: Optional[int] = None
        self.is_terminal_state: bool = False
        
        # History
        self.offer_history = []
        
    def reset(self, round_id: int, **kwargs) -> Dict[str, Any]:
        """
        Reset domain for a new negotiation round.
        
        Args:
            round_id: Round identifier
            **kwargs: Optional overrides for buyer_max, seller_min, etc.
            
        Returns:
            Dict with initial state info
        """
        self.round_id = round_id
        self.is_terminal_state = False
        self.current_offer = None
        self.agreement_price = None
        self.last_offerer_id = None
        self.offer_history = []
        
        # Standard Training Distribution (Paper Replication)
        # Buyer Max ~ N(900, 50)
        # Seller Min = Buyer Max - 500 (Fixed ZOPA width of 500)
        
        if "buyer_max_range" in kwargs:
            # Manual override path (legacy or specific tests)
            buyer_max_range = kwargs.get("buyer_max_range", (100, 200))
            self.buyer_max = round(random.uniform(*buyer_max_range), 2)
            seller_min_range = kwargs.get("seller_min_range", (50, 150))
            self.seller_min = round(random.uniform(*seller_min_range), 2)
        elif "buyer_max" in kwargs and "seller_min" in kwargs:
            # Direct override
            self.buyer_max = float(kwargs["buyer_max"])
            self.seller_min = float(kwargs["seller_min"])
        else:
            # Default to matching the training dataset distribution
            self.buyer_max = round(random.gauss(900, 50), 2)
            self.seller_min = round(self.buyer_max - 500.0, 2)
        
        return {
            "buyer_max": self.buyer_max,
            "seller_min": self.seller_min
        }

    def get_private_context(self, agent_id: int) -> Dict[str, Any]:
        """
        Get agent-specific private context.
        
        Args:
            agent_id: Agent identifier (1=buyer, 2=seller)
            
        Returns:
            Dict with role-specific information
        """
        context = {
            "history": self.offer_history,
            "last_offer": self.current_offer
        }
        if agent_id == self.buyer_id:
            context.update({
                "role": "buyer",
                "max_willingness_to_pay": self.buyer_max,
                "profit_formula": "Utility = Max_Willingness - Price"
            })
            return context
        elif agent_id == self.seller_id:
            context.update({
                "role": "seller",
                "min_acceptable_price": self.seller_min,
                "profit_formula": "Utility = Price - Min_Acceptable"
            })
            return context
        return {}

    def get_public_context(self) -> Dict[str, Any]:
        """Get publicly visible context."""
        return {
            "item_name": "Widget",
            "currency": "USD"
        }

    def parse_agent_action(self, agent_id: int, text: str) -> ParsedAction:
        """
        Parse agent text response into a structured action.
        
        Args:
            agent_id: Agent making the action
            text: Raw text response
            
        Returns:
            ParsedAction with type, content, and raw text
        """
        text_upper = text.upper()
        has_offer_keyword = "OFFER" in text_upper
        has_accept_keyword = ("ACCEPT" in text_upper or "AGREE" in text_upper)
        
        # 1. Try explicit OFFER pattern
        offer_pattern = r"OFFER\s+\$?(\d+(?:\.\d+)?)"
        match = re.search(offer_pattern, text, re.IGNORECASE)
        
        if match:
            try:
                price = float(match.group(1))
                return ParsedAction("OFFER", price, text)
            except ValueError:
                pass

        # 2. Check for ACCEPT (without explicit OFFER)
        if has_accept_keyword and not match:
            return ParsedAction("ACCEPT", None, text)

        # 3. Try implicit price pattern ($X)
        price_pattern = r"\$\s?(\d+(?:\.\d+)?)"
        matches = re.findall(price_pattern, text)
        if matches:
            unique_prices = list(set([float(m) for m in matches]))
            if len(unique_prices) == 1:
                return ParsedAction("OFFER", unique_prices[0], text)
            elif len(unique_prices) > 0:
                return ParsedAction("OFFER", float(matches[-1]), text)
            
        return ParsedAction("INVALID", None, text)

    def is_valid_action(self, action: ParsedAction) -> bool:
        """Check if an action is valid in current state."""
        if action.action_type == "OFFER":
            return isinstance(action.offer_content, (int, float)) and action.offer_content >= 0
        
        if action.action_type == "ACCEPT":
            return self.current_offer is not None
            
        return False

    def apply_action(self, action: ParsedAction, agent_id: int) -> bool:
        """
        Apply an action to the domain state.
        
        Args:
            action: The parsed action
            agent_id: Agent making the action
            
        Returns:
            True if action was applied successfully
        """
        if not self.is_valid_action(action):
            return False
            
        if action.action_type == "OFFER":
            self.current_offer = action.offer_content
            self.last_offerer_id = agent_id
            role = "buyer" if agent_id == self.buyer_id else "seller"
            self.offer_history.append((role, self.current_offer))
            
        elif action.action_type == "ACCEPT":
            if self.last_offerer_id is not None and self.last_offerer_id != agent_id:
                self.agreement_price = self.current_offer
                self.is_terminal_state = True
                
        return True

    def is_agreement(self) -> bool:
        """Check if agreement has been reached."""
        return self.agreement_price is not None

    def get_outcome(self) -> Dict[str, Any]:
        """
        Get negotiation outcome.
        
        Returns:
            Dict with agreement status, utilities, price, and ZOPA check
        """
        if not self.agreement_price:
            return {
                "agreement": False,
                "agent1_utility": 0.0,
                "agent2_utility": 0.0,
                "price": None,
                "within_zopa": False
            }
            
        # ZOPA Check: Seller Min <= Price <= Buyer Max
        within_zopa = (self.seller_min <= self.agreement_price <= self.buyer_max)
            
        return {
            "agreement": True,
            "agent1_utility": self.buyer_max - self.agreement_price,  # Buyer utility
            "agent2_utility": self.agreement_price - self.seller_min,  # Seller utility
            "price": self.agreement_price,
            "within_zopa": within_zopa
        }

    def format_agent_prompt_context(self, agent_id: int) -> str:
        """
        Format context for agent prompt.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Formatted string for agent prompt
        """
        ctx = self.get_private_context(agent_id)
        role = ctx.get("role", "negotiator")
        
        # ZOPA info for both agents
        zopa_info = f"\n## ZONE OF POSSIBLE AGREEMENT (ZOPA):\n"
        zopa_info += f"- Seller's minimum acceptable price: ${self.seller_min}\n"
        zopa_info += f"- Buyer's maximum willingness to pay: ${self.buyer_max}\n"
        zopa_info += f"- Agreement is only possible between ${self.seller_min} and ${self.buyer_max}.\n"
        
        msg = f"\nYou are the {role}.\n"
        msg += zopa_info
        
        if role == "buyer":
            msg += f"\n## YOUR GOAL:\n"
            msg += f"Get the LOWEST price possible. Your limit is ${self.buyer_max}.\n"
            msg += f"Start near ${self.seller_min} (the low end) and increase gradually.\n"
        else:
            msg += f"\n## YOUR GOAL:\n"
            msg += f"Get the HIGHEST price possible. Your limit is ${self.seller_min}.\n"
            msg += f"Start near ${self.buyer_max} (the high end) and decrease gradually.\n"
            
        return msg

    def get_zopa(self) -> Dict[str, Any]:
        """Get ZOPA information."""
        return {
            "exists": self.seller_min <= self.buyer_max,
            "low": self.seller_min,
            "high": self.buyer_max,
            "width": max(0, self.buyer_max - self.seller_min)
        }
