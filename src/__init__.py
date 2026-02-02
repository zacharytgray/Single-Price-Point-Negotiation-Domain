"""
Single Price Point Negotiation Domain - Source Package

A comprehensive framework for single-issue price negotiation research,
featuring HyperLoRA-based Janus agents, deterministic strategies,
and training/evaluation infrastructure.
"""

__version__ = "1.0.0"

from src.core.price_structures import PriceAction, PriceState
from src.domain.single_issue_price_domain import SingleIssuePriceDomain
from src.agents.agent_factory import create_agent

__all__ = [
    'PriceAction',
    'PriceState', 
    'SingleIssuePriceDomain',
    'create_agent'
]
