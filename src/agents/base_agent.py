"""
Base agent class that defines the interface for all negotiation agents.
All agent types inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseAgent(ABC):
    """
    Abstract base class for all negotiation agents.
    Defines the interface that all agents must implement.
    """
    
    def __init__(self, agent_id: int, model_name: str, system_instructions_file: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.system_instructions_file = system_instructions_file
        self.memory = []
        self.domain_private_context = {}
        self.domain_public_context = {}
        
    @abstractmethod
    async def generate_response(self) -> str:
        """
        Generate a response during negotiation.
        
        Returns:
            str: The agent's response message
        """
        pass
    
    @abstractmethod
    def add_to_memory(self, role: str, content: str):
        """
        Add a message to the agent's memory.
        
        Args:
            role: Role of the message (system, user, assistant)
            content: Content of the message
        """
        pass
    
    @abstractmethod
    def reset_memory(self):
        """
        Reset the agent's memory to initial state (system instructions only).
        """
        pass
    
    def set_domain_context(self, private_: Dict, public_: Dict):
        """
        Set the domain context for this negotiation round.
        
        Args:
            private_: Private context for this agent
            public_: Public context shared by all agents
        """
        self.domain_private_context = private_
        self.domain_public_context = public_

    def propose_action(self, state: Any) -> Any:
        """
        Produce a structured action for the given state. 
        Only used in dataset mode (no LLM).
        
        Args:
            state: The current state object (e.g. PriceState)
            
        Returns:
            The proposed action object (e.g. PriceAction)
        """
        raise NotImplementedError("This agent does not support structured action proposal.")

    def get_agent_context(self) -> str:
        """
        Get the context string for this agent.
        
        Returns:
            str: Formatted context string
        """
        if self.domain_private_context:
            return str(self.domain_private_context)
        return ""
    
    def should_make_deterministic_proposal(self, turn_number: int = 1) -> bool:
        """
        Check if this agent should make a deterministic proposal.
        Override in deterministic agent types.
        
        Args:
            turn_number: Current turn number
        
        Returns:
            bool: True if agent should make deterministic proposal
        """
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata for logging.
        
        Returns:
            Dict with agent information
        """
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "agent_type": self.__class__.__name__
        }
