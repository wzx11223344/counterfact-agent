"""
Evidence grading and sourcing for counterfactual claims.

Grades evidence quality on a structured A/B/C/D scale based on:
- Verifiability: Can the claim be independently verified?
- Source credibility: How trustworthy is the source?
- Methodological rigor: How sound is the method producing the evidence?
- Corroboration: Is the claim supported by multiple independent sources?

Combines rule-based heuristics with LLM evaluation for nuanced cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Grade definitions
# ---------------------------------------------------------------------------

class EvidenceGrade(str, Enum):
    """Evidence quality grade."""

    A = "A"  # Well-verified, multiple strong sources, rigorous method
    B = "B"  # Reasonable evidence, some corroboration, standard method
    C = "C"  # Weak evidence, single source, unverified method
    D = "D"  # Unsupported, speculative, or contradictory evidence


GRADE_DESCRIPTIONS = {
    EvidenceGrade.A: "Strong evidence: multiple independent, credible sources; rigorous methodology; directly verifiable.",
    EvidenceGrade.B: "Moderate evidence: reasonable source with some corroboration; standard methodology; partially verifiable.",
    EvidenceGrade.C: "Weak evidence: single or low-credibility source; ad-hoc methodology; difficult to verify.",
    EvidenceGrade.D: "Unsupported: no credible source; speculative reasoning; contradictory evidence exists.",
}


@dataclass
class GradedClaim:
    """A single claim with its evidence grade."""

    claim: str
    grade: EvidenceGrade
    source: str
    rationale: str
    confidence_tag: str = ""  # HIGH_CONF, MED_CONF, LOW_CONF, SPECULATIVE


@dataclass
class EvidenceReport:
    """Complete evidence grading report."""

    claims: List[GradedClaim]
    overall_grade: EvidenceGrade
    flagged_unsupported: List[str]  # claims with grade D
    source_consistency: Dict[str, List[str]]  # source -> list of claims
    cross_validation_notes: str


# ---------------------------------------------------------------------------
# Rule-based heuristics
# ---------------------------------------------------------------------------

# Indicators of high-quality sources
_HIGH_QUALITY_DOMAINS = {
    "gov", "edu", "who.int", "imf.org", "worldbank.org", "oecd.org",
    "nber.org", "aeaweb.org", "repec.org", "arxiv.org",
}

# Indicators of lower-quality sources
_LOW_QUALITY_INDICATORS = [
    r"opinion", r"blog", r"forum", r"reddit", r"twitter", r"x\.com",
    r"medium\.com", r"substack\.com",
]

# Strong methodological terms
_STRONG_METHOD_TERMS = [
    r"randomi[sz]ed\s+control", r"difference.in.difference", r"regression\s+discontinuity",
    r"instrumental\s+variable", r"synthetic\s+control", r"propensity\s+score",
    r"fixed\s+effects", r"panel\s+data", r"peer.reviewed", r"published\s+in",
]

# Weak methodological terms
_WEAK_METHOD_TERMS = [
    r"anecdotal", r"correlation\s+is\s+not\s+causation", r"preliminary",
    r"unpublished", r"working\s+paper", r"pre.print",
]


class EvidenceGrader:
    """
    Grade evidence quality for counterfactual claims.

    Uses both rule-based heuristics and LLM evaluation to assign
    structured A/B/C/D grades to each claim.

    Parameters
    ----------
    client : OpenAI or None
        OpenAI client for LLM-based grading. If None, only heuristic grading
        is used.
    model : str
        Model name for LLM grading.
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o",
    ) -> None:
        self._client = client
        self._model = model

    # ------------------------------------------------------------------
    # Single claim grading
    # ------------------------------------------------------------------

    def grade_evidence(
        self,
        source: str,
        claim: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> GradedClaim:
        """
        Grade a single claim's evidence quality.

        Parameters
        ----------
        source : str
            The source of the claim (URL, publication name, dataset reference).
        claim : str
            The claim text.
        context : dict, optional
            Additional context about the analysis.

        Returns
        -------
        GradedClaim
        """
        # Step 1: Heuristic pre-scoring
        heuristic_grade, heuristic_reason = self._heuristic_grade(source, claim)

        # Step 2: LLM refinement (if available)
        if self._client is not None:
            llm_grade, llm_reason = self._llm_grade(source, claim, context)
            # Blend: LLM can override heuristic up or down one level
            final_grade, final_reason = self._blend_grades(
                heuristic_grade, heuristic_reason, llm_grade, llm_reason
            )
        else:
            final_grade, final_reason = heuristic_grade, heuristic_reason

        return GradedClaim(
            claim=claim,
            grade=final_grade,
            source=source,
            rationale=final_reason,
        )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        claims: List[str],
        sources: List[str],
    ) -> Dict[str, List[str]]:
        """
        Check for source consistency across claims.

        Identifies which sources support which claims and flags
        potential contradictions.

        Parameters
        ----------
        claims : list of str
            Claim texts.
        sources : list of str
            Source identifiers, same order as claims.

        Returns
        -------
        dict
            Mapping from source to list of related claim indices.
        """
        # Group claims by source domain
        source_map: Dict[str, List[str]] = {}
        for i, (claim, source) in enumerate(zip(claims, sources)):
            domain = self._extract_domain(source)
            if domain not in source_map:
                source_map[domain] = []
            source_map[domain].append(f"[{i}] {claim[:120]}...")

        return source_map

    # ------------------------------------------------------------------
    # Flag unsupported claims
    # ------------------------------------------------------------------

    def flag_unsupported(self, claims: List[GradedClaim]) -> List[str]:
        """
        Find claims without adequate evidence support (grade D).

        Parameters
        ----------
        claims : list of GradedClaim

        Returns
        -------
        list of str
            Claim texts flagged as unsupported.
        """
        return [c.claim for c in claims if c.grade == EvidenceGrade.D]

    # ------------------------------------------------------------------
    # Batch grading
    # ------------------------------------------------------------------

    def grade_all(
        self,
        claims: List[str],
        sources: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """
        Grade all claims and produce a comprehensive evidence report.

        Parameters
        ----------
        claims : list of str
        sources : list of str
        context : dict, optional

        Returns
        -------
        EvidenceReport
        """
        if len(claims) != len(sources):
            raise ValueError(
                f"claims and sources must have same length: "
                f"{len(claims)} vs {len(sources)}"
            )

        # Grade each claim
        graded = [
            self.grade_evidence(src, clm, context)
            for clm, src in zip(claims, sources)
        ]

        # Flag unsupported
        unsupported = self.flag_unsupported(graded)

        # Cross-validate
        source_consistency = self.cross_validate(claims, sources)

        # Overall grade: worst grade among all claims
        grade_values = {
            EvidenceGrade.A: 4, EvidenceGrade.B: 3,
            EvidenceGrade.C: 2, EvidenceGrade.D: 1,
        }
        overall = min(graded, key=lambda g: grade_values[g.grade]).grade if graded else EvidenceGrade.D

        return EvidenceReport(
            claims=graded,
            overall_grade=overall,
            flagged_unsupported=unsupported,
            source_consistency=source_consistency,
            cross_validation_notes=self._generate_cross_val_notes(graded, source_consistency),
        )

    # ------------------------------------------------------------------
    # Heuristic grading
    # ------------------------------------------------------------------

    def _heuristic_grade(self, source: str, claim: str) -> Tuple[EvidenceGrade, str]:
        """Apply rule-based heuristics to grade evidence."""
        score = 2  # Start at C
        reasons: List[str] = []

        # Source credibility
        domain = self._extract_domain(source)
        if any(hqd in domain.lower() for hqd in _HIGH_QUALITY_DOMAINS):
            score += 1
            reasons.append("high-quality source domain")
        if any(re.search(pat, source.lower()) for pat in _LOW_QUALITY_INDICATORS):
            score -= 1
            reasons.append("low-quality source indicator")

        # Methodological rigor
        if any(re.search(pat, claim.lower()) for pat in _STRONG_METHOD_TERMS):
            score += 1
            reasons.append("strong methodological signal")
        if any(re.search(pat, claim.lower()) for pat in _WEAK_METHOD_TERMS):
            score -= 1
            reasons.append("weak methodological signal")

        # Specificity: vague claims get penalized
        vague_patterns = [r"may\s+have", r"could\s+be", r"might\s+be", r"possibly"]
        vague_count = sum(1 for p in vague_patterns if re.search(p, claim.lower()))
        if vague_count >= 3:
            score -= 1
            reasons.append("excessive hedging / vagueness")

        # Numerical specificity
        if re.search(r"\d+%|\d+\.\d+", claim):
            score += 1
            reasons.append("numerical specificity")

        # Clamp and convert
        if score >= 4:
            grade = EvidenceGrade.A
        elif score == 3:
            grade = EvidenceGrade.B
        elif score == 2:
            grade = EvidenceGrade.C
        else:
            grade = EvidenceGrade.D

        reason_text = "; ".join(reasons) if reasons else "insufficient signals"
        return grade, reason_text

    # ------------------------------------------------------------------
    # LLM grading
    # ------------------------------------------------------------------

    def _llm_grade(
        self,
        source: str,
        claim: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[EvidenceGrade, str]:
        """Use LLM to evaluate evidence quality."""
        if self._client is None:
            return EvidenceGrade.C, "no LLM available"

        prompt = f"""Evaluate the evidence quality for this claim.

Source: {source}
Claim: {claim}
Context: {context or "None"}

Grade on a scale of A/B/C/D:
- A: Strong evidence -- multiple credible sources, rigorous methodology, verifiable
- B: Moderate evidence -- reasonable source, some corroboration, standard method
- C: Weak evidence -- single source, ad-hoc method, difficult to verify
- D: Unsupported -- no credible source, speculative, contradictory evidence

Return ONLY a JSON object: {{"grade": "A", "rationale": "..."}}"""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are an evidence quality evaluator. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            import json
            data = json.loads(response.choices[0].message.content or "{}")
            grade = EvidenceGrade(data.get("grade", "C"))
            rationale = data.get("rationale", "LLM evaluation failed")
            return grade, rationale
        except Exception:
            return EvidenceGrade.C, "LLM evaluation error"

    @staticmethod
    def _blend_grades(
        heuristic_grade: EvidenceGrade,
        heuristic_reason: str,
        llm_grade: EvidenceGrade,
        llm_reason: str,
    ) -> Tuple[EvidenceGrade, str]:
        """Blend heuristic and LLM grades."""
        grade_values = {EvidenceGrade.A: 4, EvidenceGrade.B: 3, EvidenceGrade.C: 2, EvidenceGrade.D: 1}
        h_val = grade_values[heuristic_grade]
        l_val = grade_values[llm_grade]

        # Average and round
        avg = (h_val + l_val) / 2
        rounded = round(avg)
        clamped = max(1, min(4, rounded))
        final = [EvidenceGrade.D, EvidenceGrade.C, EvidenceGrade.B, EvidenceGrade.A][clamped - 1]

        combined_reason = f"heuristic[{heuristic_grade}]: {heuristic_reason} | llm[{llm_grade}]: {llm_reason}"
        return final, combined_reason

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_domain(source: str) -> str:
        """Extract domain from a URL or return source as-is."""
        match = re.search(r"https?://([^/]+)", source)
        return match.group(1) if match else source

    @staticmethod
    def _generate_cross_val_notes(
        graded: List[GradedClaim],
        source_map: Dict[str, List[str]],
    ) -> str:
        """Generate cross-validation summary notes."""
        parts: List[str] = []

        # Count sources per grade
        a_count = sum(1 for g in graded if g.grade == EvidenceGrade.A)
        d_count = sum(1 for g in graded if g.grade == EvidenceGrade.D)

        parts.append(f"Total claims: {len(graded)}")
        parts.append(f"Grade A claims: {a_count}")
        parts.append(f"Grade D claims: {d_count}")
        parts.append(f"Unique sources: {len(source_map)}")

        if d_count > len(graded) * 0.5:
            parts.append("WARNING: More than 50% of claims are unsupported (grade D).")

        return "\n".join(parts)
