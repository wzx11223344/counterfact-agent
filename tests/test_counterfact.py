"""
Tests for the counterfact-agent package.

Covers the synthetic control engine, evidence grader, reporter, and
end-to-end pipeline (quantitative only, no LLM required for tests).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package is importable from the test directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from counterfact_agent.evidence import EvidenceGrade, EvidenceGrader, GradedClaim
from counterfact_agent.reporter import AnalysisResult, Reporter
from counterfact_agent.synthetic_control import (
    DonorUnit,
    SyntheticControl,
    SyntheticControlResult,
)


# ---------------------------------------------------------------------------
# SyntheticControl tests
# ---------------------------------------------------------------------------

class TestSyntheticControl(unittest.TestCase):
    """Tests for the SyntheticControl class."""

    def setUp(self) -> None:
        self.sc = SyntheticControl()
        self.rng = np.random.default_rng(42)
        self.pre_period = (0, 10)
        self.post_period = (10, 20)
        self.n_periods = 20
        self.n_units = 10

    def _make_units(self, with_treatment: bool = True) -> tuple[DonorUnit, list[DonorUnit]]:
        """Create a target unit and candidate donors."""
        times = np.arange(self.n_periods, dtype=float)
        t0 = self.pre_period[1]

        # Target: linear trend, plus treatment effect if requested
        target_outcomes = 100.0 + 2.0 * times + self.rng.normal(0, 1.0, self.n_periods)
        if with_treatment:
            target_outcomes[t0:] += 10.0  # treatment effect of +10

        target = DonorUnit(
            name="target",
            covariates=np.array([1.0, 0.5]),
            outcomes=target_outcomes,
        )

        # Candidates: similar trend, no treatment
        candidates = []
        for i in range(self.n_units):
            outcomes = 100.0 + 2.0 * times + self.rng.normal(0, 1.5, self.n_periods)
            # Add some heterogeneity
            outcomes += self.rng.uniform(-5, 5)
            candidates.append(DonorUnit(
                name=f"donor_{i}",
                covariates=np.array([1.0 + self.rng.uniform(-0.2, 0.2), 0.5 + self.rng.uniform(-0.1, 0.1)]),
                outcomes=outcomes,
            ))

        return target, candidates

    def test_donor_pool_construction(self) -> None:
        """Donor pool should select units with sufficient correlation."""
        target, candidates = self._make_units()
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period,
            min_correlation=0.5, max_donors=5,
        )
        self.assertGreater(len(donors), 0)
        self.assertLessEqual(len(donors), 5)

    def test_donor_pool_empty_when_no_correlation(self) -> None:
        """Donor pool should be empty if no donors correlate."""
        target, candidates = self._make_units()

        # Make all candidates completely uncorrelated
        for c in candidates:
            c.outcomes = self.rng.normal(0, 1, self.n_periods)

        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period,
            min_correlation=0.99,  # impossibly high
        )
        self.assertEqual(len(donors), 0)

    def test_weight_optimization(self) -> None:
        """Weights should sum to 1 and be non-negative."""
        target, candidates = self._make_units()
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period,
            min_correlation=0.3, max_donors=10,
        )

        weights, obj_val = self.sc.compute_synthetic_weights(
            target, donors, self.pre_period, n_restarts=5,
        )
        self.assertEqual(len(weights), len(donors))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        self.assertTrue(np.all(weights >= -1e-10))  # allow tiny floating errors

    def test_weight_optimization_empty_donors(self) -> None:
        """Empty donor pool returns empty weights."""
        target, _ = self._make_units()
        weights, obj_val = self.sc.compute_synthetic_weights(
            target, [], self.pre_period,
        )
        self.assertEqual(len(weights), 0)

    def test_counterfactual_prediction(self) -> None:
        """Counterfactual should be weighted average of donors."""
        target, candidates = self._make_units()
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period, max_donors=5,
        )
        weights, _ = self.sc.compute_synthetic_weights(
            target, donors, self.pre_period,
        )

        donor_mat = np.array([u.outcomes for u in donors])
        cf = self.sc.predict_counterfactual(
            target, weights, donor_mat, self.post_period,
        )
        self.assertEqual(len(cf), self.post_period[1] - self.post_period[0])

    def test_pre_treatment_rmse(self) -> None:
        """Pre-treatment RMSE should be computed correctly."""
        target, candidates = self._make_units()
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period, max_donors=5,
        )
        weights, _ = self.sc.compute_synthetic_weights(
            target, donors, self.pre_period,
        )
        rmse = self.sc.rmse_pre_weight(target, weights, donors, self.pre_period)
        self.assertGreater(rmse, 0)

    def test_counterfactual_gap_detects_treatment(self) -> None:
        """The gap should reflect the treatment effect."""
        target, candidates = self._make_units(with_treatment=True)
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period, max_donors=5,
        )
        weights, _ = self.sc.compute_synthetic_weights(
            target, donors, self.pre_period,
        )
        donor_mat = np.array([u.outcomes for u in donors])
        cf = self.sc.predict_counterfactual(target, weights, donor_mat, self.post_period)
        gap = self.sc.counterfactual_gap(target, cf, self.post_period)

        # The gap should be positive (treatment effect is +10)
        self.assertGreater(float(np.mean(gap)), 0)

    def test_no_treatment_no_gap(self) -> None:
        """Without treatment, the gap should be close to zero."""
        target, candidates = self._make_units(with_treatment=False)
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period, max_donors=5,
        )
        weights, _ = self.sc.compute_synthetic_weights(
            target, donors, self.pre_period,
        )
        donor_mat = np.array([u.outcomes for u in donors])
        cf = self.sc.predict_counterfactual(target, weights, donor_mat, self.post_period)
        gap = self.sc.counterfactual_gap(target, cf, self.post_period)

        # Gap should be small (no treatment)
        self.assertLess(abs(float(np.mean(gap))), 3.0)

    def test_placebo_tests(self) -> None:
        """Placebo tests should produce valid ratios and p-value."""
        target, candidates = self._make_units(with_treatment=True)
        donors = self.sc.build_donor_pool(
            target, candidates, self.pre_period, max_donors=5,
        )

        placebo_gaps, placebo_ratios, target_ratio, p_value = self.sc.compute_placebo_tests(
            target, donors, self.pre_period, self.post_period,
        )

        self.assertGreater(len(placebo_gaps), 0)
        self.assertGreater(len(placebo_ratios), 0)
        self.assertGreater(target_ratio, 0)
        if p_value is not None:
            self.assertGreaterEqual(p_value, 0.0)
            self.assertLessEqual(p_value, 1.0)

    def test_full_pipeline(self) -> None:
        """Full SC pipeline should return a complete result."""
        target, candidates = self._make_units(with_treatment=True)

        result = self.sc.run(
            target=target,
            candidates=candidates,
            pre_period=self.pre_period,
            post_period=self.post_period,
            min_correlation=0.3,
            max_donors=5,
        )

        self.assertIsInstance(result, SyntheticControlResult)
        self.assertGreater(len(result.donor_names), 0)
        self.assertGreater(len(result.weights), 0)
        self.assertTrue(result.convergence)
        self.assertIsNotNone(result.inference_p_value)

    def test_full_pipeline_no_donors(self) -> None:
        """Pipeline with impossible correlation threshold returns empty result."""
        target, candidates = self._make_units()

        result = self.sc.run(
            target=target,
            candidates=candidates,
            pre_period=self.pre_period,
            post_period=self.post_period,
            min_correlation=0.999,  # impossibly high
        )

        self.assertEqual(len(result.donor_names), 0)
        self.assertFalse(result.convergence)


# ---------------------------------------------------------------------------
# EvidenceGrader tests
# ---------------------------------------------------------------------------

class TestEvidenceGrader(unittest.TestCase):
    """Tests for the EvidenceGrader class (heuristic mode, no LLM needed)."""

    def setUp(self) -> None:
        self.grader = EvidenceGrader(client=None)  # heuristic-only

    def test_grade_high_quality_source(self) -> None:
        """gov domain with numerical specificity should score well."""
        result = self.grader.grade_evidence(
            source="https://www.census.gov/trade/data",
            claim="US imports from China decreased by 12.5% in 2019 according to randomized control trial data published in peer-reviewed journal.",
        )
        self.assertIn(result.grade, (EvidenceGrade.A, EvidenceGrade.B))

    def test_grade_low_quality_source(self) -> None:
        """Blog source with vague claims should score poorly."""
        result = self.grader.grade_evidence(
            source="https://someblog.medium.com/opinion-piece",
            claim="The tariff may have possibly affected trade somehow, and it could be that some firms were affected.",
        )
        self.assertIn(result.grade, (EvidenceGrade.C, EvidenceGrade.D))

    def test_flag_unsupported(self) -> None:
        """Flag unsupported should find grade D claims."""
        claims = [
            GradedClaim(claim="Good claim", grade=EvidenceGrade.A, source="gov", rationale="strong"),
            GradedClaim(claim="Bad claim", grade=EvidenceGrade.D, source="blog", rationale="weak"),
            GradedClaim(claim="OK claim", grade=EvidenceGrade.B, source="edu", rationale="moderate"),
        ]
        unsupported = self.grader.flag_unsupported(claims)
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0], "Bad claim")

    def test_cross_validate(self) -> None:
        """Cross-validation should group claims by source domain."""
        claims = ["Claim A", "Claim B", "Claim C"]
        sources = [
            "https://www.census.gov/data",
            "https://www.census.gov/other",
            "https://example.com/blog",
        ]
        result = self.grader.cross_validate(claims, sources)
        self.assertIn("www.census.gov", result)
        self.assertIn("example.com", result)

    def test_grade_all(self) -> None:
        """Batch grading should produce a complete report."""
        claims = ["Verified trade decline of 12%", "Possible minor impact"]
        sources = ["https://www.census.gov/trade", "https://medium.com/opinion"]
        report = self.grader.grade_all(claims, sources)
        self.assertEqual(len(report.claims), 2)
        self.assertIsInstance(report.overall_grade, EvidenceGrade)


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------

class TestReporter(unittest.TestCase):
    """Tests for the Reporter class."""

    def setUp(self) -> None:
        self.reporter = Reporter(title_prefix="Test Analysis")

    def _make_sample_result(self) -> AnalysisResult:
        """Create a minimal AnalysisResult for testing."""
        return AnalysisResult(
            event_description="Test policy event: minimum wage increased by 20%",
            event_date="2024-01-01",
            sc_result=None,
            mechanism_narrative=None,
            confidence_discussion=None,
            policy_implications="[MED_CONF] Policymakers should consider phased implementation.",
            evidence_report=None,
        )

    def test_full_report_generation(self) -> None:
        """Full report should be a non-empty string with expected sections."""
        result = self._make_sample_result()
        report = self.reporter.generate_full_report(result)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)
        self.assertIn("Test Analysis", report)
        self.assertIn("Test policy event", report)
        self.assertIn("Policy Implications", report)

    def test_executive_brief(self) -> None:
        """Executive brief should be concise."""
        result = self._make_sample_result()
        brief = self.reporter.executive_brief(result)
        self.assertIsInstance(brief, str)
        self.assertIn("Executive Brief", brief)
        self.assertIn("Bottom Line", brief)

    def test_sensitivity_analysis(self) -> None:
        """Sensitivity analysis should be a valid string."""
        result = self._make_sample_result()
        sa = self.reporter.sensitivity_analysis(result)
        self.assertIsInstance(sa, str)
        self.assertIn("Sensitivity Analysis", sa)

    def test_sensitivity_with_parameters(self) -> None:
        """Sensitivity analysis with parameter sweep should include table."""
        result = self._make_sample_result()
        params = {"donor_threshold": [0.2, 0.3, 0.5]}
        sa = self.reporter.sensitivity_analysis(result, parameters=params)
        self.assertIn("Parameter Sweep", sa)
        self.assertIn("donor_threshold", sa)

    def test_html_generation(self) -> None:
        """HTML report should be a complete HTML document."""
        result = self._make_sample_result()
        html = self.reporter.generate_html(result)
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)
        self.assertIn("Counterfactual Analysis Report", html)


# ---------------------------------------------------------------------------
# Integration tests (no LLM)
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline (no LLM)."""

    def setUp(self) -> None:
        """Create a temporary CSV file with panel data."""
        self.tmpdir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.tmpdir, "test_data.csv")
        self._create_test_data()

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_data(self) -> None:
        """Create a simple panel dataset."""
        rng = np.random.default_rng(999)
        records = []
        for unit_idx in range(10):
            unit = f"unit_{unit_idx}"
            for t in range(20):
                outcome = 100.0 + 2.0 * t + rng.normal(0, 2.0)
                # Treatment for unit_0 after t=10
                if unit == "unit_0" and t >= 10:
                    outcome -= 15.0
                records.append({
                    "unit": unit,
                    "time": t,
                    "outcome": outcome,
                    "gdp": rng.uniform(100, 500),
                })
        df = pd.DataFrame(records)
        df.to_csv(self.data_path, index=False)

    def test_engine_no_llm(self) -> None:
        """Engine should work without an API key (quantitative only)."""
        from counterfact_agent import CounterfactualEngine

        engine = CounterfactualEngine(openai_api_key="")

        result = engine.analyze(
            event_description="Test tariff imposed on unit_0",
            pre_data=self.data_path,
            covariates=["gdp"],
            target_name="unit_0",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.event_description, "Test tariff imposed on unit_0")
        # Without LLM, narrative should be None
        self.assertIsNone(result.mechanism_narrative)

    def test_engine_with_data(self) -> None:
        """Engine with data should produce SC results."""
        from counterfact_agent import CounterfactualEngine

        engine = CounterfactualEngine(openai_api_key="")

        result = engine.analyze(
            event_description="Test policy",
            pre_data=self.data_path,
            covariates=["gdp"],
            target_name="unit_0",
            pre_period=(0, 10),
            post_period=(10, 20),
        )

        self.assertIsNotNone(result.sc_result)
        sc = result.sc_result
        self.assertGreater(len(sc.donor_names), 0)
        self.assertTrue(sc.convergence)

        # The gap should be negative (treatment reduced outcome)
        if len(sc.gap) > 0:
            self.assertLess(float(np.mean(sc.gap)), 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
