"""
Agent factory for creating different types of negotiation agents.
"""

from typing import Dict, Any, Type, Optional
from src.agents.base_agent import BaseAgent
from src.agents.basic_price_agent import BasicPriceAgent
from src.agents.price_strategy_agent import PriceStrategyWrapperAgent
from src.agents.price_strategies import STRATEGY_REGISTRY, DeterministicPriceAgent
from config.settings import DEBUG_MODE, PRICE_SYSTEM_INSTRUCTIONS_FILE, PRICE_DETERMINISTIC_INSTRUCTIONS_FILE


def create_agent(
    agent_type: str,
    role: str,
    strategy: Optional[str] = None,
    model_name: Optional[str] = None,
    janus_adapter_path: Optional[str] = None,
    janus_model_path: Optional[str] = None,
    rho: Optional[float] = None
) -> Any:
    """
    Helper function to create an agent instance with simpler arguments.
    Matches usage in negotiation.py.
    """
    agent_id = 1 if role == "buyer" else 2
    
    # specialized handling for deterministic agents
    if agent_type == "deterministic":
        if not strategy:
            raise ValueError("Strategy must be provided for deterministic agent")
        return AgentFactory.create_deterministic_agent(agent_id, strategy)
    
    # specialized handling for janus
    if agent_type == "janus":
        if not janus_adapter_path:
            raise ValueError("Adapter path needed for Janus")
        # Janus factory logic expects 'model_path' and 'adapter_path' keywords
        return AgentFactory.create_agent(
            "janus", 
            agent_id, 
            model_name=janus_model_path or "Qwen/Qwen2-7B-Instruct",
            system_instructions_file="", # Not used
            adapter_path=janus_adapter_path,
            rho=rho if rho is not None else 0.5
        )

    # default fallback for LLM agents
    instructions = PRICE_SYSTEM_INSTRUCTIONS_FILE
    if strategy:
        # LLM wrapping a strategy (PriceStrategyWrapperAgent)
        return AgentFactory.create_agent(
            "price_strategy",
            agent_id,
            model_name=model_name,
            system_instructions_file=instructions,
            strategy_name=strategy
        )
    
    # Generic LLM agent
    return AgentFactory.create_agent(
        agent_type,
        agent_id,
        model_name=model_name,
        system_instructions_file=instructions
    )


class AgentFactory:
    """
    Factory class for creating different types of negotiation agents.
    """
    
    # Registry of available agent types
    AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
        "basic": BasicPriceAgent,
        "basic_price": BasicPriceAgent,
        "price_strategy": PriceStrategyWrapperAgent,
    }
    
    @classmethod
    def create_agent(cls, agent_type: str, agent_id: int, model_name: str,
                     system_instructions_file: str, **kwargs) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Unique identifier for this agent (1 or 2)
            model_name: Name of the LLM model to use
            system_instructions_file: Path to system instructions file
            **kwargs: Additional parameters specific to the agent type
            
        Returns:
            BaseAgent: Instance of the specified agent type
        """
        # Handle Janus agent specially (lazy import to avoid torch dependency issues)
        if agent_type == "janus":
            from src.agents.janus_agent import JanusAgent
            role = "buyer" if agent_id == 1 else "seller"
            hf_path = kwargs.get("model_path", model_name)
            
            return JanusAgent(
                agent_id, role,
                model_path=hf_path,
                adapter_path=kwargs.get("adapter_path", "checkpoints/janus_v1/final"),
                rho=kwargs.get("rho", 0.5)
            )
        
        # Handle deterministic strategies
        if agent_type in STRATEGY_REGISTRY:
            # Create a PriceStrategyWrapperAgent with this strategy
            kwargs["strategy_name"] = agent_type
            return PriceStrategyWrapperAgent(
                agent_id, model_name, system_instructions_file, **kwargs
            )
        
        # Handle common mappings
        if agent_type == "boulware":
            kwargs.setdefault("strategy_name", "boulware_conceding")
            return PriceStrategyWrapperAgent(
                agent_id, model_name, system_instructions_file, **kwargs
            )
        
        if agent_type == "tit_for_tat":
            kwargs["strategy_name"] = "tit_for_tat"
            return PriceStrategyWrapperAgent(
                agent_id, model_name, system_instructions_file, **kwargs
            )
        
        # Standard agent types
        if agent_type not in cls.AGENT_TYPES:
            available_types = ", ".join(list(cls.AGENT_TYPES.keys()) + list(STRATEGY_REGISTRY.keys()))
            raise ValueError(f"Unknown agent type '{agent_type}'. Available types: {available_types}")
        
        agent_class = cls.AGENT_TYPES[agent_type]

        if DEBUG_MODE:
            print(
                f"[DEBUG][AgentFactory] type={agent_type} id={agent_id} model={model_name} "
                f"kwargs={kwargs}"
            )

        return agent_class(agent_id, model_name, system_instructions_file, **kwargs)
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: Type[BaseAgent]):
        """Register a new agent type with the factory."""
        cls.AGENT_TYPES[agent_type] = agent_class
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available agent types."""
        return list(cls.AGENT_TYPES.keys()) + list(STRATEGY_REGISTRY.keys())
    
    @classmethod
    def create_deterministic_agent(cls, agent_id: int, strategy_name: str,
                                    strategy_params: Optional[Dict] = None) -> DeterministicPriceAgent:
        """
        Create a pure deterministic agent (no LLM).
        For dataset generation mode.
        """
        return DeterministicPriceAgent(agent_id, strategy_name, strategy_params)


class AgentConfig:
    """Configuration helper for different agent types."""
    
    @staticmethod
    def basic_config() -> Dict[str, Any]:
        return {}
    
    @staticmethod
    def strategy_config(strategy_name: str, **params) -> Dict[str, Any]:
        return {"strategy_name": strategy_name, **params}
    
    @staticmethod
    def janus_config(rho: float = 0.5, adapter_path: str = "checkpoints/janus_v1/final") -> Dict[str, Any]:
        return {"rho": rho, "adapter_path": adapter_path}
    
    @staticmethod
    def get_config_for_type(agent_type: str) -> Dict[str, Any]:
        """Get default configuration for an agent type."""
        if agent_type in ["basic", "basic_price"]:
            return AgentConfig.basic_config()
        elif agent_type in STRATEGY_REGISTRY:
            return AgentConfig.strategy_config(agent_type)
        elif agent_type == "janus":
            return AgentConfig.janus_config()
        return {}
