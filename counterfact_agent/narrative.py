"""
LLM-powered narrative reasoning for counterfactual analysis.

This module generates structured qualitative explanations of WHY estimated
gaps exist, using mechanism-level reasoning about direct effects, spillovers,
equilibrium adjustments, and heterogeneous responses. Every output includes
explicit confidence tags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Confidence tags
# ---------------------------------------------------------------------------

CONFIDENCE_TAGS = {
    "HIGH_CONF":     "Multiple independent sources corroborate; quantitative evidence available.",
    "MED_CONF":      "Supported by theory or partial evidence; reasonable inference.",
    "LOW_CONF":      "Plausible but speculative; limited direct evidence.",
    "SPECULATIVE":   "Theoretically possible; no direct evidence available.",
}

CONFIDENCE_ORDER = {"HIGH_CONF": 3, "MED_CONF": 2, "LOW_CONF": 1, "SPECULATIVE": 0}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MechanismNarrative:
    """Structured explanation of causal mechanisms."""

    direct_effects: str
    spillover_effects: str
    equilibrium_adjustments: str
    heterogeneous_responses: str
    synthesis: str
    confidence_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConfidenceDiscussion:
    """Explicit discussion of uncertainty around estimates."""

    overall_confidence: str  # HIGH_CONF / MED_CONF / LOW_CONF / SPECULATIVE
    quantitative_uncertainty: str
    qualitative_uncertainty: str
    key_caveats: List[str]
    robustness_notes: str


@dataclass
class NarrativeOutput:
    """Complete narrative generation result."""

    mechanism: MechanismNarrative
    confidence_discussion: ConfidenceDiscussion
    policy_implications: str
    raw_response: str


# ---------------------------------------------------------------------------
# Narrative generator
# ---------------------------------------------------------------------------

class NarrativeGenerator:
    """
    Generate structured counterfactual narratives using an LLM.

    Uses temperature=0.2 for factual consistency. Every output includes
    explicit [HIGH_CONF] / [MED_CONF] / [LOW_CONF] / [SPECULATIVE] tags.

    Parameters
    ----------
    client : OpenAI
        Configured OpenAI client instance (or compatible).
    model : str
        Model name to use (default: "gpt-4o").
    temperature : float
        LLM temperature. Kept low (0.2) for factual consistency.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Mechanism narrative
    # ------------------------------------------------------------------

    def generate_mechanism_narrative(
        self,
        event: str,
        counterfactuals: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> MechanismNarrative:
        """
        Generate a structured mechanism-level explanation for the observed gap.

        Explains WHY the gap between observed and counterfactual outcomes exists,
        covering direct effects, spillovers, equilibrium adjustments, and
        heterogeneous responses.

        Parameters
        ----------
        event : str
            Description of the policy event.
        counterfactuals : dict
            Dictionary containing estimated gaps, donor information, placebo
            results, etc.
        context : dict, optional
            Additional contextual information.

        Returns
        -------
        MechanismNarrative
        """
        prompt = self._build_mechanism_prompt(event, counterfactuals, context)
        response = self._call_llm(prompt)
        return self._parse_mechanism_response(response)

    def _build_mechanism_prompt(
        self,
        event: str,
        counterfactuals: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Construct the mechanism narrative prompt."""
        context_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "None"
        cf_str = json.dumps(counterfactuals, indent=2, ensure_ascii=False)

        return f"""You are a policy counterfactual reasoning expert. Given a policy event, observed outcomes, and estimated counterfactual outcomes, construct a mechanism-level explanation of WHY the gap exists.

## Policy Event
{event}

## Estimated Counterfactuals
{cf_str}

## Additional Context
{context_str}

## Instructions

Structure your response as a JSON object with the following fields. For each claim, tag confidence as [HIGH_CONF], [MED_CONF], [LOW_CONF], or [SPECULATIVE].

### direct_effects (string)
Explain the direct, first-order effects of the policy. Who was directly affected? What immediate behavioral or economic responses occurred? Be specific about channels.

### spillover_effects (string)
Explain indirect or spillover effects: impacts on related markets, sectors, or populations not directly targeted by the policy.

### equilibrium_adjustments (string)
Explain how the system adjusted toward a new equilibrium. How did supply, demand, prices, or incentives re-calibrate over time?

### heterogeneous_responses (string)
Explain variation in responses across different groups, regions, or time periods. Were some actors more affected than others? Why?

### synthesis (string)
Synthesize the four mechanisms into a coherent narrative. What is the overall explanation for the estimated treatment effect?

### confidence_map (object)
A mapping from each section name to the overall confidence level for claims in that section: "HIGH_CONF", "MED_CONF", "LOW_CONF", or "SPECULATIVE".

Output ONLY valid JSON, no markdown fences."""

    def _parse_mechanism_response(self, raw: str) -> MechanismNarrative:
        """Parse LLM JSON response into a MechanismNarrative."""
        try:
            data = self._extract_json(raw)
            return MechanismNarrative(
                direct_effects=data.get("direct_effects", ""),
                spillover_effects=data.get("spillover_effects", ""),
                equilibrium_adjustments=data.get("equilibrium_adjustments", ""),
                heterogeneous_responses=data.get("heterogeneous_responses", ""),
                synthesis=data.get("synthesis", ""),
                confidence_map=data.get("confidence_map", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Graceful fallback: treat raw text as synthesis
            return MechanismNarrative(
                direct_effects="[SPECULATIVE] Unable to parse structured response.",
                spillover_effects="",
                equilibrium_adjustments="",
                heterogeneous_responses="",
                synthesis=raw,
                confidence_map={"synthesis": "LOW_CONF"},
            )

    # ------------------------------------------------------------------
    # Confidence discussion
    # ------------------------------------------------------------------

    def generate_confidence_discussion(
        self,
        gap: Any,
        placebo_results: Optional[Dict[str, Any]],
        evidence_quality: Dict[str, str],
    ) -> ConfidenceDiscussion:
        """
        Generate a structured discussion of uncertainty.

        Parameters
        ----------
        gap : Any
            The estimated treatment gap (could be float, array, or dict).
        placebo_results : dict or None
            Results from placebo testing.
        evidence_quality : dict
            Evidence grades by claim.

        Returns
        -------
        ConfidenceDiscussion
        """
        gap_str = str(gap)
        placebo_str = json.dumps(placebo_results, indent=2) if placebo_results else "None"
        ev_str = json.dumps(evidence_quality, indent=2)

        prompt = f"""You are an uncertainty quantification expert. Given counterfactual estimates and supporting evidence, produce a structured confidence discussion.

## Estimated Gap
{gap_str}

## Placebo Test Results
{placebo_str}

## Evidence Quality by Claim
{ev_str}

## Instructions

Return a JSON object with these fields:

### overall_confidence (string)
One of: "HIGH_CONF", "MED_CONF", "LOW_CONF", "SPECULATIVE". Consider both quantitative precision and qualitative evidence strength.

### quantitative_uncertainty (string)
Discuss: magnitude uncertainty, sensitivity to specifications, assumptions about the model. How much could the point estimate change under reasonable alternatives?

### qualitative_uncertainty (string)
Discuss: strength of the causal narrative, alternative explanations, confounding factors that could produce similar patterns.

### key_caveats (array of strings)
List the 3-5 most important caveats a policymaker should know.

### robustness_notes (string)
How robust do the findings appear? What additional data or analysis would strengthen or weaken confidence?

Output ONLY valid JSON, no markdown fences."""

        response = self._call_llm(prompt)
        try:
            data = self._extract_json(response)
            return ConfidenceDiscussion(
                overall_confidence=data.get("overall_confidence", "LOW_CONF"),
                quantitative_uncertainty=data.get("quantitative_uncertainty", ""),
                qualitative_uncertainty=data.get("qualitative_uncertainty", ""),
                key_caveats=data.get("key_caveats", []),
                robustness_notes=data.get("robustness_notes", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return ConfidenceDiscussion(
                overall_confidence="LOW_CONF",
                quantitative_uncertainty=response,
                qualitative_uncertainty="",
                key_caveats=[],
                robustness_notes="",
            )

    # ------------------------------------------------------------------
    # Policy implications
    # ------------------------------------------------------------------

    def generate_policy_implications(
        self,
        narrative: MechanismNarrative,
        gap_estimate: Any,
    ) -> str:
        """
        Generate policy implications from the counterfactual analysis.

        Translates findings into actionable insights for policymakers.

        Parameters
        ----------
        narrative : MechanismNarrative
            The mechanism narrative.
        gap_estimate : Any
            The estimated treatment effect.

        Returns
        -------
        str
            Policy implications text with confidence tags.
        """
        prompt = f"""You are a policy advisor. Based on the following counterfactual analysis, produce actionable policy implications.

## Mechanism Narrative
- Direct Effects: {narrative.direct_effects}
- Spillover Effects: {narrative.spillover_effects}
- Equilibrium Adjustments: {narrative.equilibrium_adjustments}
- Heterogeneous Responses: {narrative.heterogeneous_responses}
- Synthesis: {narrative.synthesis}

## Estimated Treatment Effect
{gap_estimate}

## Instructions

Write a concise policy implications section (3-5 paragraphs). For each implication:
1. State the key finding
2. Explain its policy relevance
3. Suggest a specific actionable recommendation
4. Tag confidence: [HIGH_CONF], [MED_CONF], [LOW_CONF], or [SPECULATIVE]

Be specific and avoid generic advice. Reference the mechanisms above to ground recommendations in the analysis."""
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous policy counterfactual reasoning expert. "
                        "Always tag claims with confidence levels. Be specific, "
                        "evidence-based, and avoid vague generalizations. "
                        "When uncertain, say so explicitly rather than hedging."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_json(raw: str) -> Dict[str, Any]:
        """Extract JSON from a string that may have surrounding text."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find JSON between braces
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise json.JSONDecodeError("No JSON found", raw, 0)
