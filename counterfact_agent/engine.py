"""
Main counterfactual analysis pipeline.

Orchestrates the full analysis: event framing, donor pool construction,
synthetic control fitting, narrative generation, evidence grading, and
report assembly. NOT just an LLM wrapper -- runs actual synthetic control
math where data is available, falling back to structured qualitative
reasoning only when quantitative data is insufficient.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from counterfact_agent.evidence import EvidenceGrader, EvidenceReport
from counterfact_agent.narrative import (
    ConfidenceDiscussion,
    MechanismNarrative,
    NarrativeGenerator,
)
from counterfact_agent.reporter import AnalysisResult, Reporter
from counterfact_agent.synthetic_control import (
    DonorUnit,
    SyntheticControl,
    SyntheticControlResult,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for the CounterfactualEngine."""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    min_donor_correlation: float = 0.3
    max_donors: int = 20
    n_restarts: int = 10
    lambda_v: Optional[np.ndarray] = None
    fallback_to_qualitative: bool = True


class CounterfactualEngine:
    """
    Policy counterfactual reasoning engine.

    Given a policy event, this engine:
    1. Frames the event and identifies the treatment
    2. Constructs a donor pool from candidate units
    3. Fits synthetic control weights (quantitative)
    4. Generates mechanism narratives (LLM)
    5. Grades evidence quality
    6. Assembles a structured report

    The key differentiator: runs actual synthetic control math where data
    is available, not just an LLM wrapper.

    Parameters
    ----------
    config : EngineConfig or dict
        Configuration. If dict, passed to EngineConfig(**kwargs).
    """

    def __init__(self, **kwargs: Any) -> None:
        if "config" in kwargs:
            self.config = kwargs["config"]
            if isinstance(self.config, dict):
                self.config = EngineConfig(**self.config)
        else:
            self.config = EngineConfig(**kwargs)

        # Initialize subsystems
        client = OpenAI(api_key=self.config.openai_api_key) if self.config.openai_api_key else None
        self._sc = SyntheticControl(lambda_v=self.config.lambda_v)
        self._narrator = NarrativeGenerator(
            client=client,
            model=self.config.openai_model,
            temperature=self.config.llm_temperature,
        ) if client else None
        self._grader = EvidenceGrader(client=client, model=self.config.openai_model)
        self._reporter = Reporter()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        event_description: str,
        pre_data: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        target_name: str = "Treated Unit",
        candidate_names: Optional[List[str]] = None,
        event_date: Optional[str] = None,
        pre_period: Optional[Tuple[int, int]] = None,
        post_period: Optional[Tuple[int, int]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Run the full counterfactual analysis pipeline.

        Parameters
        ----------
        event_description : str
            Natural language description of the policy event.
        pre_data : str, optional
            Path to CSV file containing panel data. Expected columns:
            'unit', 'time', 'outcome', plus covariate columns.
        covariates : list of str, optional
            Names of covariate columns in the data.
        target_name : str
            Name of the treated unit in the data.
        candidate_names : list of str, optional
            Names of candidate donor units. If None, all non-target units
            in the data are used.
        event_date : str, optional
            Date of the policy event (ISO format: 'YYYY-MM-DD').
        pre_period : tuple of (start, end), optional
            Time indices for the pre-treatment period.
        post_period : tuple of (start, end), optional
            Time indices for the post-treatment period.
        context : dict, optional
            Additional contextual information for narrative generation.

        Returns
        -------
        AnalysisResult
            Complete result including SC output, narratives, evidence grades.
        """
        # Step 1: Event framing
        framed_event = self._frame_event(event_description)

        # Step 2: Load data and construct donor pool
        sc_result: Optional[SyntheticControlResult] = None
        if pre_data is not None:
            try:
                sc_result = self._run_synthetic_control(
                    pre_data=pre_data,
                    covariates=covariates or [],
                    target_name=target_name,
                    candidate_names=candidate_names,
                    pre_period=pre_period,
                    post_period=post_period,
                )
            except Exception as e:
                warnings.warn(
                    f"Synthetic control failed: {e}. "
                    f"Falling back to qualitative analysis."
                )
                if not self.config.fallback_to_qualitative:
                    raise

        # Step 3: Narrative generation (LLM)
        mechanism_narrative: Optional[MechanismNarrative] = None
        confidence_discussion: Optional[ConfidenceDiscussion] = None
        policy_implications: str = ""
        if self._narrator is not None:
            cf_dict = self._sc_result_to_dict(sc_result)
            mechanism_narrative = self._narrator.generate_mechanism_narrative(
                framed_event, cf_dict, context
            )
            # Extract all claims from the narrative for evidence grading
            narrative_claims = self._extract_narrative_claims(mechanism_narrative)
            evidence_quality = self._build_evidence_map(narrative_claims)

            confidence_discussion = self._narrator.generate_confidence_discussion(
                gap=cf_dict.get("gap", "N/A"),
                placebo_results=cf_dict.get("placebo", {}),
                evidence_quality=evidence_quality,
            )

            policy_implications = self._narrator.generate_policy_implications(
                mechanism_narrative, cf_dict.get("gap", "N/A")
            )

        # Step 4: Evidence grading
        evidence_report: Optional[EvidenceReport] = None
        if self._narrator is not None and mechanism_narrative is not None:
            claims_list, sources_list = self._collect_claims_and_sources(
                mechanism_narrative, context or {}
            )
            evidence_report = self._grader.grade_all(claims_list, sources_list, context)

        # Step 5: Build result
        result = AnalysisResult(
            event_description=event_description,
            event_date=event_date,
            sc_result=sc_result,
            mechanism_narrative=mechanism_narrative,
            confidence_discussion=confidence_discussion,
            policy_implications=policy_implications,
            evidence_report=evidence_report,
            metadata={
                "model": self.config.openai_model,
                "donor_correlation_threshold": self.config.min_donor_correlation,
                "max_donors": self.config.max_donors,
                "llm_temperature": self.config.llm_temperature,
                "quantitative_available": sc_result is not None and sc_result.convergence,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Report generation shortcuts
    # ------------------------------------------------------------------

    def full_report(self, result: AnalysisResult) -> str:
        """Generate a comprehensive Markdown report."""
        return self._reporter.generate_full_report(result)

    def executive_brief(self, result: AnalysisResult) -> str:
        """Generate a 2-page executive brief."""
        return self._reporter.executive_brief(result)

    def sensitivity_report(
        self,
        result: AnalysisResult,
        parameters: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """Generate a sensitivity analysis report."""
        return self._reporter.sensitivity_analysis(result, parameters)

    def html_report(self, result: AnalysisResult) -> str:
        """Generate an interactive HTML report."""
        return self._reporter.generate_html(result)

    # ------------------------------------------------------------------
    # Internal: Synthetic Control pipeline
    # ------------------------------------------------------------------

    def _run_synthetic_control(
        self,
        pre_data: str,
        covariates: List[str],
        target_name: str,
        candidate_names: Optional[List[str]],
        pre_period: Optional[Tuple[int, int]],
        post_period: Optional[Tuple[int, int]],
    ) -> SyntheticControlResult:
        """Load data, build units, and run SC."""
        df = pd.read_csv(pre_data)

        # Infer unit and time columns
        unit_col = self._find_column(df, ["unit", "region", "country", "state", "id"])
        time_col = self._find_column(df, ["time", "year", "period", "date", "t"])
        outcome_col = self._find_column(df, ["outcome", "y", "value", "target", "dependent"])

        # Pivot to panel: index=unit, columns=time, values=outcome
        outcomes_pivot = df.pivot(index=unit_col, columns=time_col, values=outcome_col)
        time_periods = outcomes_pivot.columns.tolist()
        n_periods = len(time_periods)

        if pre_period is None:
            pre_period = (0, n_periods // 2)
        if post_period is None:
            post_period = (n_periods // 2, n_periods)

        # Build target DonorUnit
        target_outcomes = outcomes_pivot.loc[target_name].values.astype(float)
        target_covs = self._extract_unit_covariates(df, target_name, covariates, unit_col)
        target = DonorUnit(name=target_name, covariates=target_covs, outcomes=target_outcomes)

        # Build candidate DonorUnits
        if candidate_names is None:
            candidate_names = [u for u in outcomes_pivot.index if u != target_name]

        candidates: List[DonorUnit] = []
        for name in candidate_names:
            if name not in outcomes_pivot.index:
                continue
            unit_outcomes = outcomes_pivot.loc[name].values.astype(float)
            unit_covs = self._extract_unit_covariates(df, name, covariates, unit_col)
            candidates.append(DonorUnit(name=name, covariates=unit_covs, outcomes=unit_outcomes))

        return self._sc.run(
            target=target,
            candidates=candidates,
            pre_period=pre_period,
            post_period=post_period,
            covariates=covariates,
            min_correlation=self.config.min_donor_correlation,
            max_donors=self.config.max_donors,
            n_restarts=self.config.n_restarts,
        )

    # ------------------------------------------------------------------
    # Internal: Event framing
    # ------------------------------------------------------------------

    def _frame_event(self, description: str) -> str:
        """Frame the event description with structured context."""
        # For now, return the description as-is. Future versions could use
        # LLM to extract structured event metadata.
        return description

    # ------------------------------------------------------------------
    # Internal: Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
        """Find the first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        # Fallback: first column
        return df.columns[0]

    @staticmethod
    def _extract_unit_covariates(
        df: pd.DataFrame,
        unit_name: str,
        covariates: List[str],
        unit_col: str,
    ) -> np.ndarray:
        """Extract average covariate values for a unit."""
        unit_data = df[df[unit_col] == unit_name]
        if not covariates:
            return np.array([])
        cov_values = []
        for cov in covariates:
            if cov in df.columns:
                cov_values.append(float(unit_data[cov].mean()))
        return np.array(cov_values)

    # ------------------------------------------------------------------
    # Internal: Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sc_result_to_dict(sc: Optional[SyntheticControlResult]) -> Dict[str, Any]:
        """Convert SC result to a JSON-serializable dict for LLM prompts."""
        if sc is None:
            return {"status": "not_available", "reason": "insufficient_quantitative_data"}

        return {
            "status": "computed",
            "target": sc.target_name,
            "donors": sc.donor_names,
            "weights": {name: float(w) for name, w in zip(sc.donor_names, sc.weights) if w > 0.001},
            "pre_rmse": float(sc.pre_rmse),
            "convergence": sc.convergence,
            "gap": {
                "mean": float(np.mean(sc.gap)) if len(sc.gap) > 0 else None,
                "median": float(np.median(sc.gap)) if len(sc.gap) > 0 else None,
                "cumulative": float(np.sum(sc.gap)) if len(sc.gap) > 0 else None,
            },
            "placebo": {
                "target_ratio": float(sc.target_ratio),
                "p_value": float(sc.inference_p_value) if sc.inference_p_value is not None else None,
                "n_placebos": len(sc.placebo_ratios),
            },
        }

    @staticmethod
    def _extract_narrative_claims(narrative: MechanismNarrative) -> List[str]:
        """Extract individual claims from a mechanism narrative."""
        claims: List[str] = []
        for field in ["direct_effects", "spillover_effects", "equilibrium_adjustments",
                       "heterogeneous_responses", "synthesis"]:
            text = getattr(narrative, field, "")
            if text:
                # Split on sentence boundaries
                for sentence in text.replace("\n", " ").split(". "):
                    s = sentence.strip().rstrip(".")
                    if len(s) > 20:
                        claims.append(s)
        return claims

    @staticmethod
    def _build_evidence_map(claims: List[str]) -> Dict[str, str]:
        """Build a simple evidence quality map for claims."""
        return {claim[:80]: "Not graded" for claim in claims}

    @staticmethod
    def _collect_claims_and_sources(
        narrative: MechanismNarrative,
        context: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """Collect claims and their sources for evidence grading."""
        claims_list: List[str] = []
        sources_list: List[str] = []

        narrative_text = " ".join([
            narrative.direct_effects,
            narrative.spillover_effects,
            narrative.equilibrium_adjustments,
            narrative.heterogeneous_responses,
            narrative.synthesis,
        ])

        for sentence in narrative_text.replace("\n", " ").split(". "):
            s = sentence.strip().rstrip(".")
            if len(s) > 30:
                claims_list.append(s)
                sources_list.append(context.get("source", "LLM-generated narrative"))

        return claims_list, sources_list
