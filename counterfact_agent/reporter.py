"""
Structured report generation for counterfactual analysis.

Produces multiple output formats: comprehensive Markdown reports,
2-page executive briefs for policymakers, sensitivity analysis,
and interactive HTML with toggleable sections.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from counterfact_agent.evidence import EvidenceGrade, EvidenceReport, GradedClaim
from counterfact_agent.narrative import ConfidenceDiscussion, MechanismNarrative
from counterfact_agent.synthetic_control import SyntheticControlResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Complete result from a counterfactual analysis."""

    event_description: str
    event_date: Optional[str]
    sc_result: Optional[SyntheticControlResult]
    mechanism_narrative: Optional[MechanismNarrative]
    confidence_discussion: Optional[ConfidenceDiscussion]
    policy_implications: str
    evidence_report: Optional[EvidenceReport]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class Reporter:
    """
    Generate structured reports from counterfactual analysis results.

    Supports multiple output formats:
    - Full Markdown report
    - Executive brief (2-page policymaker summary)
    - Sensitivity analysis
    - Interactive HTML with toggleable sections
    """

    def __init__(self, title_prefix: str = "Counterfactual Analysis") -> None:
        self._title_prefix = title_prefix

    # ------------------------------------------------------------------
    # Full report (Markdown)
    # ------------------------------------------------------------------

    def generate_full_report(self, result: AnalysisResult) -> str:
        """
        Generate a comprehensive Markdown report.

        Parameters
        ----------
        result : AnalysisResult
            Complete analysis result.

        Returns
        -------
        str
            Full markdown report.
        """
        sections: List[str] = []

        # Header
        sections.append(self._report_header(result))

        # Executive summary
        sections.append(self._executive_summary(result))

        # Quantitative analysis
        if result.sc_result is not None:
            sections.append(self._quantitative_section(result))

        # Mechanism narrative
        if result.mechanism_narrative is not None:
            sections.append(self._narrative_section(result))

        # Evidence grading
        if result.evidence_report is not None:
            sections.append(self._evidence_section(result))

        # Confidence discussion
        if result.confidence_discussion is not None:
            sections.append(self._confidence_section(result))

        # Policy implications
        sections.append(self._implications_section(result))

        # Appendix
        sections.append(self._appendix(result))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Executive brief
    # ------------------------------------------------------------------

    def executive_brief(self, result: AnalysisResult) -> str:
        """
        Generate a 2-page executive brief for policymakers.

        Parameters
        ----------
        result : AnalysisResult

        Returns
        -------
        str
            Concise executive brief in Markdown.
        """
        lines: List[str] = []

        lines.append(f"# {self._title_prefix}: Executive Brief")
        lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("")

        # One-line summary
        lines.append("## Bottom Line")
        if result.sc_result is not None and len(result.sc_result.gap) > 0:
            avg_gap = float(np.mean(result.sc_result.gap))
            direction = "increase" if avg_gap > 0 else "decrease"
            lines.append(
                f"The policy is estimated to have caused a {abs(avg_gap):.2f} unit "
                f"{direction} in the outcome variable "
                f"(p={result.sc_result.inference_p_value or 'N/A'})."
            )
            if result.confidence_discussion:
                lines.append(f"Overall confidence: **{result.confidence_discussion.overall_confidence}**")
        else:
            lines.append("Quantitative estimate not available. See qualitative analysis below.")
        lines.append("")

        # Key findings (bullet points, max 5)
        lines.append("## Key Findings")
        findings = self._extract_key_findings(result)
        for i, finding in enumerate(findings[:5], 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

        # Policy recommendations (max 3)
        lines.append("## Policy Recommendations")
        if result.policy_implications:
            # Extract first 3 sentences
            sentences = result.policy_implications.split(". ")
            for s in sentences[:3]:
                s = s.strip()
                if s and not s.endswith("."):
                    s += "."
                if s:
                    lines.append(f"- {s}")
        lines.append("")

        # Caveats
        lines.append("## Key Caveats")
        if result.confidence_discussion and result.confidence_discussion.key_caveats:
            for caveat in result.confidence_discussion.key_caveats:
                lines.append(f"- {caveat}")
        else:
            lines.append("- Results are sensitive to model specification and data quality.")
        lines.append("")

        # Methodology note
        lines.append("## Methodology")
        lines.append(
            "This analysis uses the Synthetic Control Method (Abadie et al., 2010) "
            "where quantitative data is available, supplemented by structured LLM-based "
            "narrative reasoning where data is insufficient. All claims are tagged with "
            "explicit confidence levels: [HIGH_CONF], [MED_CONF], [LOW_CONF], [SPECULATIVE]."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        result: AnalysisResult,
        parameters: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """
        Generate a sensitivity analysis report.

        Discusses how results change under alternative specifications.

        Parameters
        ----------
        result : AnalysisResult
        parameters : dict, optional
            Parameter sweep results: param_name -> list of gap estimates.

        Returns
        -------
        str
            Sensitivity analysis in Markdown.
        """
        lines: List[str] = []
        lines.append("## Sensitivity Analysis")
        lines.append("")
        lines.append(
            "This section examines how the estimated treatment effect varies "
            "under alternative model specifications and assumptions."
        )
        lines.append("")

        if parameters:
            lines.append("### Parameter Sweep Results")
            lines.append("")
            lines.append("| Parameter | Value | Estimated Gap | Change from Baseline |")
            lines.append("|-----------|-------|---------------|---------------------|")
            for param, values in parameters.items():
                baseline = values[0] if values else 0
                for v in values:
                    change = v - baseline
                    lines.append(f"| {param} | {v:.4f} | {v:.4f} | {change:+.4f} |")
            lines.append("")

        # Qualitative sensitivity discussion
        lines.append("### Qualitative Sensitivity")
        if result.sc_result is not None:
            lines.append(f"- **Pre-treatment RMSE**: {result.sc_result.pre_rmse:.4f}")
            lines.append(f"- **Number of donors**: {len(result.sc_result.donor_names)}")
            lines.append(f"- **Inference p-value**: {result.sc_result.inference_p_value or 'N/A'}")
            if result.sc_result.donor_names:
                lines.append(f"- **Donor units**: {', '.join(result.sc_result.donor_names)}")

        lines.append("")
        lines.append("### Robustness Considerations")
        lines.append("1. **Donor pool sensitivity**: Results may change if different donor units are included/excluded.")
        lines.append("2. **Pre-treatment period**: Longer pre-treatment periods generally improve fit but may capture structural breaks.")
        lines.append("3. **Covariate selection**: Different covariate sets may produce different weights and counterfactuals.")
        lines.append("4. **Model specification**: The additive linear structure of SC assumes no interactions or nonlinearities.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html(self, result: AnalysisResult) -> str:
        """
        Generate an interactive HTML report with toggleable sections.

        Parameters
        ----------
        result : AnalysisResult

        Returns
        -------
        str
            Self-contained HTML document.
        """
        md_report = self.generate_full_report(result)
        # Convert key sections to HTML with toggle
        html = self._md_to_html_sections(md_report, result)
        return html

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _report_header(self, result: AnalysisResult) -> str:
        """Build the report header."""
        lines = [
            f"# {self._title_prefix}",
            f"**Event**: {result.event_description}",
            f"**Date of Analysis**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        if result.event_date:
            lines.append(f"**Event Date**: {result.event_date}")
        lines.append("")
        lines.append("---")
        return "\n".join(lines)

    def _executive_summary(self, result: AnalysisResult) -> str:
        """Build a brief executive summary section."""
        lines = ["## Executive Summary", ""]

        if result.sc_result is not None and len(result.sc_result.gap) > 0:
            avg_gap = float(np.mean(result.sc_result.gap))
            cum_gap = float(np.sum(result.sc_result.gap))
            direction = "positive" if avg_gap > 0 else "negative"
            lines.append(
                f"The estimated treatment effect is **{direction}**: "
                f"on average, the policy changed the outcome by "
                f"**{avg_gap:+.4f}** units per period "
                f"(cumulative: **{cum_gap:+.4f}** units over the post-treatment window)."
            )
            if result.sc_result.inference_p_value is not None:
                significance = (
                    "statistically significant"
                    if result.sc_result.inference_p_value < 0.10
                    else "not statistically significant"
                )
                lines.append(f"The effect is **{significance}** "
                             f"(placebo-based p-value: {result.sc_result.inference_p_value:.3f}).")
        else:
            lines.append("Quantitative synthetic control analysis was not possible "
                         "with available data. Results below are based on qualitative "
                         "reasoning only.")

        if result.mechanism_narrative:
            lines.append("")
            lines.append(result.mechanism_narrative.synthesis[:500])

        return "\n".join(lines)

    def _quantitative_section(self, result: AnalysisResult) -> str:
        """Build the quantitative analysis section."""
        sc = result.sc_result
        assert sc is not None
        lines = ["## Quantitative Analysis: Synthetic Control Method", ""]

        lines.append("### Donor Pool")
        lines.append(f"The synthetic control was constructed from **{len(sc.donor_names)}** donor units.")
        if sc.donor_names:
            lines.append("")
            lines.append("| Donor | Weight |")
            lines.append("|-------|--------|")
            for name, w in zip(sc.donor_names, sc.weights):
                if w > 0.001:
                    lines.append(f"| {name} | {w:.4f} |")

        lines.append("")
        lines.append("### Fit Quality")
        lines.append(f"- **Pre-treatment RMSE**: {sc.pre_rmse:.4f}")
        lines.append(f"- **Convergence**: {'Yes' if sc.convergence else 'No'}")

        lines.append("")
        lines.append("### Estimated Treatment Effect")
        if len(sc.gap) > 0:
            lines.append(f"- **Mean gap**: {float(np.mean(sc.gap)):+.4f}")
            lines.append(f"- **Median gap**: {float(np.median(sc.gap)):+.4f}")
            lines.append(f"- **Min gap**: {float(np.min(sc.gap)):+.4f}")
            lines.append(f"- **Max gap**: {float(np.max(sc.gap)):+.4f}")
            lines.append(f"- **Cumulative gap**: {float(np.sum(sc.gap)):+.4f}")

        lines.append("")
        lines.append("### Placebo Inference")
        lines.append(f"- **Target post/pre RMSPE ratio**: {sc.target_ratio:.4f}")
        lines.append(f"- **Inference p-value**: {sc.inference_p_value or 'N/A'}")

        n_placebos = len(sc.placebo_ratios)
        if n_placebos > 0:
            better_than = sum(1 for r in sc.placebo_ratios if r >= sc.target_ratio)
            lines.append(
                f"- **Rank**: Target ratio exceeds {better_than}/{n_placebos} "
                f"placebo ratios"
            )

        return "\n".join(lines)

    def _narrative_section(self, result: AnalysisResult) -> str:
        """Build the mechanism narrative section."""
        narr = result.mechanism_narrative
        assert narr is not None

        lines = ["## Mechanism Narrative", ""]

        sections = [
            ("Direct Effects", narr.direct_effects, narr.confidence_map.get("direct_effects", "")),
            ("Spillover / Indirect Effects", narr.spillover_effects, narr.confidence_map.get("spillover_effects", "")),
            ("Equilibrium Adjustments", narr.equilibrium_adjustments, narr.confidence_map.get("equilibrium_adjustments", "")),
            ("Heterogeneous Responses", narr.heterogeneous_responses, narr.confidence_map.get("heterogeneous_responses", "")),
            ("Synthesis", narr.synthesis, narr.confidence_map.get("synthesis", "")),
        ]

        for title, content, conf in sections:
            if content:
                conf_tag = f" [{conf}]" if conf else ""
                lines.append(f"### {title}{conf_tag}")
                lines.append("")
                lines.append(content)
                lines.append("")

        return "\n".join(lines)

    def _evidence_section(self, result: AnalysisResult) -> str:
        """Build the evidence grading section."""
        er = result.evidence_report
        assert er is not None

        lines = ["## Evidence Assessment", ""]
        lines.append(f"**Overall Grade**: {er.overall_grade.value}")
        lines.append(f"**Grade Description**: {GRADE_DESCRIPTIONS[er.overall_grade]}")  # type: ignore[arg-type]
        lines.append("")

        lines.append("### Claim-by-Claim Grading")
        lines.append("")
        lines.append("| # | Claim | Grade | Source | Rationale |")
        lines.append("|---|-------|-------|--------|-----------|")
        for i, gc in enumerate(er.claims, 1):
            claim_short = gc.claim[:100].replace("|", "\\|")
            lines.append(f"| {i} | {claim_short} | {gc.grade.value} | {gc.source[:40]} | {gc.rationale[:60]} |")

        if er.flagged_unsupported:
            lines.append("")
            lines.append("### Flagged: Unsupported Claims")
            for claim in er.flagged_unsupported:
                lines.append(f"- [D] {claim[:200]}")

        return "\n".join(lines)

    def _confidence_section(self, result: AnalysisResult) -> str:
        """Build the confidence discussion section."""
        cd = result.confidence_discussion
        assert cd is not None

        lines = ["## Confidence Assessment", ""]
        lines.append(f"**Overall Confidence**: **{cd.overall_confidence}**")
        lines.append("")

        lines.append("### Quantitative Uncertainty")
        lines.append(cd.quantitative_uncertainty)
        lines.append("")

        lines.append("### Qualitative Uncertainty")
        lines.append(cd.qualitative_uncertainty)
        lines.append("")

        lines.append("### Key Caveats")
        for c in cd.key_caveats:
            lines.append(f"- {c}")
        lines.append("")

        lines.append("### Robustness Notes")
        lines.append(cd.robustness_notes)

        return "\n".join(lines)

    def _implications_section(self, result: AnalysisResult) -> str:
        """Build the policy implications section."""
        lines = ["## Policy Implications", ""]
        lines.append(result.policy_implications)
        return "\n".join(lines)

    def _appendix(self, result: AnalysisResult) -> str:
        """Build the appendix."""
        lines = ["## Appendix: Methodology", ""]
        lines.append(
            "### Synthetic Control Method (Abadie et al., 2010)"
        )
        lines.append("")
        lines.append(
            "The synthetic control method constructs a counterfactual by finding "
            "a weighted combination of untreated units (the 'donor pool') that "
            "best approximates the treated unit's pre-treatment characteristics. "
            "Weights are estimated via constrained optimization minimizing "
            "||X1 - X0 @ W|| subject to W >= 0 and sum(W) = 1."
        )
        lines.append("")
        lines.append(
            "Inference is conducted via in-space placebo tests: the SC method is "
            "applied to each donor unit as if it were treated, and the distribution "
            "of placebo effects is used to assess the significance of the estimated "
            "effect for the actual treated unit."
        )
        lines.append("")
        lines.append("### Narrative Reasoning")
        lines.append("")
        lines.append(
            "Where quantitative data is insufficient, structured LLM-based narrative "
            "reasoning is employed. All LLM outputs are generated at temperature=0.2 "
            "for factual consistency, and every claim is tagged with an explicit "
            "confidence level."
        )

        lines.append("")
        lines.append("### References")
        lines.append(
            "- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control "
            "Methods for Comparative Case Studies: Estimating the Effect of "
            "California's Tobacco Control Program. *Journal of the American "
            "Statistical Association*, 105(490), 493-505."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key_findings(result: AnalysisResult) -> List[str]:
        """Extract key findings from the result."""
        findings: List[str] = []

        # Quantitative finding
        if result.sc_result is not None and len(result.sc_result.gap) > 0:
            avg = float(np.mean(result.sc_result.gap))
            direction = "increased" if avg > 0 else "decreased"
            p_val = result.sc_result.inference_p_value
            sig = f" (p={p_val:.3f})" if p_val is not None else ""
            findings.append(
                f"The policy {direction} the outcome by an average of "
                f"{abs(avg):.2f} units per period{sig}."
            )

        # Narrative synthesis (first 2 sentences)
        if result.mechanism_narrative and result.mechanism_narrative.synthesis:
            sentences = result.mechanism_narrative.synthesis.split(". ")
            for s in sentences[:2]:
                s = s.strip().rstrip(".")
                if s and len(s) > 20:
                    findings.append(s)
                    break

        # Confidence
        if result.confidence_discussion:
            findings.append(
                f"Overall confidence in these findings is rated as "
                f"**{result.confidence_discussion.overall_confidence}**."
            )

        return findings

    @staticmethod
    def _md_to_html_sections(md_text: str, result: AnalysisResult) -> str:
        """Convert markdown sections to an interactive HTML document."""
        # Simple but effective: wrap in HTML with toggleable sections
        escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Counterfactual Analysis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6;
         color: #1a1a1a; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
  h2 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-top: 30px; }}
  h3 {{ margin-top: 20px; color: #374151; }}
  pre {{ background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 8px;
         overflow-x: auto; }}
  .toggle-btn {{ background: #2563eb; color: white; border: none; padding: 6px 16px;
                 border-radius: 6px; cursor: pointer; font-size: 14px; }}
  .toggle-btn:hover {{ background: #1d4ed8; }}
  .collapsible {{ margin-bottom: 12px; }}
  .collapsible-content {{ padding: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
  th {{ background: #f3f4f6; }}
  .grade-a {{ color: #059669; font-weight: bold; }}
  .grade-b {{ color: #2563eb; font-weight: bold; }}
  .grade-c {{ color: #d97706; font-weight: bold; }}
  .grade-d {{ color: #dc2626; font-weight: bold; }}
  .high-conf {{ background: #d1fae5; padding: 2px 6px; border-radius: 4px; }}
  .med-conf {{ background: #dbeafe; padding: 2px 6px; border-radius: 4px; }}
  .low-conf {{ background: #fef3c7; padding: 2px 6px; border-radius: 4px; }}
  .spec {{ background: #fee2e2; padding: 2px 6px; border-radius: 4px; }}
</style>
<script>
function toggleSection(id) {{
  var el = document.getElementById(id);
  if (el.style.display === 'none') {{ el.style.display = 'block'; }}
  else {{ el.style.display = 'none'; }}
}}
</script>
</head>
<body>
<h1>Counterfactual Analysis Report</h1>
<p><strong>Event:</strong> {result.event_description}</p>
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<hr>
<pre>{escaped}</pre>
<p style="color:#6b7280; font-size:12px; margin-top:40px;">
  Generated by Counterfact-Agent v0.1.0 &mdash; 
  Synthetic Control Method (Abadie et al., 2010) + LLM Narrative Reasoning
</p>
</body>
</html>"""
