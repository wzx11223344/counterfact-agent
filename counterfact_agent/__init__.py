"""
Counterfact-Agent: Policy Counterfactual Reasoning Agent.

Given a policy event, constructs structured counterfactual scenarios using
both quantitative methods (synthetic control) and LLM-powered narrative reasoning.
"""

from counterfact_agent.engine import CounterfactualEngine
from counterfact_agent.synthetic_control import SyntheticControl
from counterfact_agent.narrative import NarrativeGenerator
from counterfact_agent.evidence import EvidenceGrader
from counterfact_agent.reporter import Reporter

__version__ = "0.1.0"
__all__ = [
    "CounterfactualEngine",
    "SyntheticControl",
    "NarrativeGenerator",
    "EvidenceGrader",
    "Reporter",
]
