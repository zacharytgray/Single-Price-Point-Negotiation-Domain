"""
Basic Price Agent - Pure LLM negotiator with no deterministic backend.
"""

from typing import Optional, Dict
from src.agents.base_agent import BaseAgent
from src.agents.ollama_agent import OllamaAgent
from config.settings import DEBUG_MODE


class BasicPriceAgent(BaseAgent):
    """
    Basic Price Agent.
    No deterministic backend. Pure LLM.
    Relies on system instructions to negotiate.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str):
        super().__init__(agent_id, model_name, system_instructions_file)
        self.ollama_agent = OllamaAgent(model_name, system_instructions_file)
        
    async def generate_response(self) -> str:
        if DEBUG_MODE:
            role = self.domain_private_context.get("role")
            max_wtp = self.domain_private_context.get("max_willingness_to_pay")
            min_price = self.domain_private_context.get("min_acceptable_price")
            last_offer = self.domain_private_context.get("last_offer")
            history = self.domain_private_context.get("history", [])
            print(
                f"[DEBUG][BasicPriceAgent] agent_id={self.agent_id} role={role} "
                f"max_wtp={max_wtp} min_price={min_price} last_offer={last_offer} "
                f"history_len={len(history)}"
            )
        return await self.ollama_agent.generate_response()
    
    def add_to_memory(self, role: str, content: str):
        self.ollama_agent.add_to_memory(role, content)
    
    def reset_memory(self):
        self.ollama_agent.reset_memory()
        
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        return False
