"""
Price Strategy Wrapper Agent - LLM wrapper around deterministic strategies.
"""

import re
from typing import Optional, Dict, Any
from src.agents.base_agent import BaseAgent
from src.agents.ollama_agent import OllamaAgent
from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from src.core.price_structures import PriceState, PriceAction


class PriceStrategyWrapperAgent(BaseAgent):
    """
    Wrapper Agent for Single Issue Price Domain.
    Uses a deterministic strategy to calculate the next move,
    then uses an LLM to wrap that move in natural language.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str,
                 strategy_name: str = "boulware_linear", strategy_params: Optional[Dict] = None):
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        
        # Use DeterministicPriceAgent logic internally
        self.det_agent = DeterministicPriceAgent(agent_id, strategy_name, strategy_params)
        
    async def generate_response(self) -> str:
        """Generate LLM response. Prompt should already have mandated offer."""
        return await self.ollama_agent.generate_response()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.add_to_memory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()
        
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        """This agent type always uses deterministic proposals."""
        return True
    
    def validate_output_matches_intent(self, response: str, intended_proposal: Dict) -> bool:
        """
        Check if the LLM's response actually contains the intended price/action.
        """
        action = intended_proposal.get("action")
        
        if action == "ACCEPT":
            return "accept" in response.lower() or "agree" in response.lower()
            
        elif action == "OFFER":
            target_price = intended_proposal.get("price")
            matches = re.findall(r'\$?\s?(\d+(?:\.\d+)?)', response)
            
            if not matches:
                return False
                
            for m in matches:
                try:
                    val = float(m)
                    if abs(val - target_price) < 1.0:  # $1 tolerance
                        return True
                except:
                    continue
            
            return False
            
        return False

    def get_deterministic_proposal(self, turn_number: int = 1) -> Optional[Dict]:
        """
        Calculate the move using the underlying strategy.
        Requires domain_private_context to be up to date.
        """
        if not self.domain_private_context:
            return None

        role = self.domain_private_context.get("role")
        history = self.domain_private_context.get("history", [])
        max_turns = self.domain_private_context.get("max_turns", 20)
        
        if role == "buyer":
            true_res = self.domain_private_context.get("max_willingness_to_pay")
        else:
            true_res = self.domain_private_context.get("min_acceptable_price")
            
        last_offer = None
        if history:
            last_offer = history[-1][1]
        
        state = PriceState(
            timestep=turn_number,
            max_turns=max_turns,
            role=role,
            last_offer_price=last_offer,
            offer_history=history,
            effective_reservation_price=true_res,
            true_reservation_price=true_res,
            public_price_range=(200, 1500)
        )
        
        action = self.det_agent.propose_action(state)
        
        result = {
            "action": action.type,
        }
        if action.type == "OFFER":
            result["price"] = action.price
            
        return result
