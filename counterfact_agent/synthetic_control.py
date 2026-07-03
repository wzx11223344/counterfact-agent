"""
Pure Python implementation of the Synthetic Control Method.

Based on: Abadie, A., Diamond, A., & Hainmueller, J. (2010).
"Synthetic Control Methods for Comparative Case Studies: Estimating the
Effect of California's Tobacco Control Program."
Journal of the American Statistical Association, 105(490), 493-505.

This module implements the full SC pipeline: donor pool construction,
constrained weight optimization, counterfactual prediction, placebo testing,
and treatment effect estimation -- all with numpy/scipy, no LLM needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DonorUnit:
    """A single donor unit with its covariates and outcome series."""

    name: str
    covariates: np.ndarray  # shape (k,) -- k predictor/covariate values
    outcomes: np.ndarray    # shape (T,) -- time series of outcome variable
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticControlResult:
    """Full output of a synthetic control analysis."""

    target_name: str
    donor_names: List[str]
    weights: np.ndarray                # learned donor weights
    pre_period: Tuple[int, int]        # (start, end) indices
    post_period: Tuple[int, int]
    pre_rmse: float                    # pre-treatment fit quality
    counterfactual: np.ndarray         # predicted outcome under no treatment
    observed: np.ndarray               # actual observed outcome
    gap: np.ndarray                    # treatment effect = observed - counterfactual
    placebo_gaps: List[np.ndarray]     # gaps for each donor placebo test
    placebo_ratios: List[float]        # post/pre RMSPE ratios
    target_ratio: float                # target unit's post/pre RMSPE ratio
    inference_p_value: Optional[float] = None
    convergence: bool = True


# ---------------------------------------------------------------------------
# Synthetic Control core
# ---------------------------------------------------------------------------

class SyntheticControl:
    """
    Pure Python implementation of the Synthetic Control Method.

    This class does NOT require an LLM. It performs the actual constrained
    optimization and counterfactual prediction using numpy/scipy.

    Parameters
    ----------
    lambda_v : np.ndarray or None
        Predictor importance weights. If None, equal weights are used.
    """

    def __init__(self, lambda_v: Optional[np.ndarray] = None) -> None:
        self._lambda_v = lambda_v

    # ------------------------------------------------------------------
    # Donor pool
    # ------------------------------------------------------------------

    def build_donor_pool(
        self,
        target: DonorUnit,
        candidates: List[DonorUnit],
        pre_period: Tuple[int, int],
        covariates: Optional[List[str]] = None,
        min_correlation: float = 0.3,
        max_donors: int = 20,
    ) -> List[DonorUnit]:
        """
        Select donor units from candidates based on pre-treatment similarity.

        Uses covariate proximity and pre-treatment outcome correlation to
        filter candidates down to a relevant donor pool.

        Parameters
        ----------
        target : DonorUnit
            The treated unit.
        candidates : list of DonorUnit
            Candidate donor units (untreated).
        pre_period : tuple of (start, end)
            Indices defining the pre-treatment window.
        covariates : list of str, optional
            Covariate names to use for matching.
        min_correlation : float
            Minimum pre-treatment outcome correlation for inclusion.
        max_donors : int
            Maximum number of donors to retain.

        Returns
        -------
        list of DonorUnit
            Selected donor pool.
        """
        if len(candidates) == 0:
            return []

        t0, t1 = pre_period
        target_pre = target.outcomes[t0:t1]
        target_covs = target.covariates

        scored: List[Tuple[float, DonorUnit]] = []
        for unit in candidates:
            # Outcome correlation in pre-period
            unit_pre = unit.outcomes[t0:t1]
            corr = self._pearson_correlation(target_pre, unit_pre)
            if corr < min_correlation:
                continue

            # Covariate distance (Mahalanobis-like if lambda_v provided)
            unit_covs = unit.covariates
            cov_dist = self._covariate_distance(target_covs, unit_covs)

            # Composite score: prioritize correlation, penalize covariate distance
            score = corr * 0.7 - cov_dist * 0.3
            scored.append((score, unit))

        # Sort by score descending, keep top max_donors
        scored.sort(key=lambda x: x[0], reverse=True)
        donors = [unit for _, unit in scored[:max_donors]]
        return donors

    # ------------------------------------------------------------------
    # Weight optimization
    # ------------------------------------------------------------------

    def compute_synthetic_weights(
        self,
        target: DonorUnit,
        donor_pool: List[DonorUnit],
        pre_period: Tuple[int, int],
        n_restarts: int = 10,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal donor weights via constrained optimization.

        Minimizes ||X1 - X0 @ W|| subject to W_i >= 0 and sum(W_i) = 1,
        where X1 are target predictors and X0 are donor predictors.

        Uses multiple random restarts to avoid local minima.

        Parameters
        ----------
        target : DonorUnit
            The treated unit.
        donor_pool : list of DonorUnit
            Selected donor units.
        pre_period : tuple of (start, end)
            Pre-treatment window.
        n_restarts : int
            Number of random restarts for optimization.

        Returns
        -------
        weights : np.ndarray of shape (J,)
            Optimal donor weights.
        objective_value : float
            Final objective value (lower is better fit).
        """
        J = len(donor_pool)
        if J == 0:
            return np.array([]), float("inf")

        t0, t1 = pre_period

        # Build predictor matrix for target
        target_pre_outcomes = target.outcomes[t0:t1]
        X1 = np.concatenate([target.covariates, target_pre_outcomes])

        # Build predictor matrix for donors
        X0_list = []
        for unit in donor_pool:
            unit_pre = unit.outcomes[t0:t1]
            X0_list.append(np.concatenate([unit.covariates, unit_pre]))
        X0 = np.column_stack(X0_list)  # shape (K, J)

        # Predictor importance weights
        if self._lambda_v is None:
            V = np.eye(X1.shape[0])
        else:
            V = np.diag(self._lambda_v)

        # Objective: (X1 - X0 @ W)' V (X1 - X0 @ W)
        def objective(w: np.ndarray) -> float:
            diff = X1 - X0 @ w
            return float(diff.T @ V @ diff)

        # Constraints: sum(w) = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: w_i >= 0
        bounds = [(0.0, 1.0) for _ in range(J)]

        best_w: Optional[np.ndarray] = None
        best_val = float("inf")

        for _ in range(n_restarts):
            # Random initial guess on simplex
            w0 = np.random.dirichlet(np.ones(J))
            res = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 5000, "ftol": 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_w = res.x

        if best_w is None:
            # Fallback: equal weights
            best_w = np.ones(J) / J
            best_val = objective(best_w)

        # Zero out negligible weights
        best_w[best_w < 1e-5] = 0.0
        best_w = best_w / best_w.sum()

        return best_w, best_val

    # ------------------------------------------------------------------
    # Counterfactual prediction
    # ------------------------------------------------------------------

    def predict_counterfactual(
        self,
        target: DonorUnit,
        weights: np.ndarray,
        donor_outcomes: np.ndarray,
        post_period: Tuple[int, int],
    ) -> np.ndarray:
        """
        Generate counterfactual predictions for the post-treatment period.

        The counterfactual is a weighted average of donor outcomes:
            Y_hat(0) = sum_j w_j * Y_jt  for t in post_period

        Parameters
        ----------
        target : DonorUnit
            The treated unit (for metadata).
        weights : np.ndarray
            Learned donor weights, shape (J,).
        donor_outcomes : np.ndarray
            Donor outcome matrix, shape (J, T).
        post_period : tuple of (start, end)

        Returns
        -------
        np.ndarray
            Predicted counterfactual outcomes for the post period.
        """
        if len(weights) == 0:
            t0, t1 = post_period
            return np.full(t1 - t0, np.nan)

        t0, t1 = post_period
        # Weighted average: (J,) @ (J, T_post) -> (T_post,)
        counterfactual = weights @ donor_outcomes[:, t0:t1]
        return counterfactual

    # ------------------------------------------------------------------
    # RMSE
    # ------------------------------------------------------------------

    def rmse_pre_weight(
        self,
        target: DonorUnit,
        weights: np.ndarray,
        donor_pool: List[DonorUnit],
        pre_period: Tuple[int, int],
    ) -> float:
        """
        Compute root mean squared error of the pre-treatment fit.

        Parameters
        ----------
        target : DonorUnit
        weights : np.ndarray
        donor_pool : list of DonorUnit
        pre_period : tuple of (start, end)

        Returns
        -------
        float
            Pre-treatment RMSE.
        """
        t0, t1 = pre_period
        actual = target.outcomes[t0:t1]
        donor_mat = np.array([u.outcomes[t0:t1] for u in donor_pool])
        predicted = weights @ donor_mat
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # ------------------------------------------------------------------
    # Treatment gap
    # ------------------------------------------------------------------

    def counterfactual_gap(
        self,
        target: DonorUnit,
        synthetic: np.ndarray,
        post_period: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute the treatment effect estimate (gap).

        gap_t = Y_observed_t - Y_counterfactual_t

        Parameters
        ----------
        target : DonorUnit
        synthetic : np.ndarray
            Counterfactual predictions for the post period.
        post_period : tuple of (start, end)

        Returns
        -------
        np.ndarray
            Gap series for the post-treatment period.
        """
        t0, t1 = post_period
        observed = target.outcomes[t0:t1]
        gap = observed - synthetic
        return gap

    # ------------------------------------------------------------------
    # Placebo tests
    # ------------------------------------------------------------------

    def compute_placebo_tests(
        self,
        target: DonorUnit,
        donor_pool: List[DonorUnit],
        pre_period: Tuple[int, int],
        post_period: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[float], float, Optional[float]]:
        """
        Run in-space placebo tests: apply SC to each donor as if treated.

        For each donor, treat it as the "target" and build a synthetic
        control from the remaining donors. Compare post/pre RMSPE ratios
        to assess whether the estimated effect for the real target is
        unusually large.

        Parameters
        ----------
        target : DonorUnit
        donor_pool : list of DonorUnit
        pre_period : tuple of (start, end)
        post_period : tuple of (start, end)

        Returns
        -------
        placebo_gaps : list of np.ndarray
            Gap estimates for each placebo test.
        placebo_ratios : list of float
            Post/pre RMSPE ratio for each placebo.
        target_ratio : float
            Post/pre RMSPE ratio for the real target.
        p_value : float or None
            Rank-based inference p-value.
        """
        t0_pre, t1_pre = pre_period
        t0_post, t1_post = post_period

        # --- Target unit ratio ---
        target_pre = target.outcomes[t0_pre:t1_pre]
        target_post = target.outcomes[t0_post:t1_post]
        pre_rmse_target = np.sqrt(np.mean(target_pre ** 2))
        post_rmse_target = np.sqrt(np.mean(target_post ** 2))
        target_ratio = post_rmse_target / pre_rmse_target if pre_rmse_target > 0 else float("inf")

        # --- Run placebo for each donor ---
        placebo_gaps: List[np.ndarray] = []
        placebo_ratios: List[float] = []

        for i, placebo_target in enumerate(donor_pool):
            # Remaining donors = all except this one
            other_donors = [d for j, d in enumerate(donor_pool) if j != i]
            if len(other_donors) < 2:
                continue

            try:
                w, _ = self.compute_synthetic_weights(
                    placebo_target, other_donors, pre_period, n_restarts=3
                )
                donor_mat = np.array([u.outcomes for u in other_donors])
                cf = self.predict_counterfactual(placebo_target, w, donor_mat, post_period)
                gap = self.counterfactual_gap(placebo_target, cf, post_period)
                placebo_gaps.append(gap)

                # Ratio
                placebo_pre = placebo_target.outcomes[t0_pre:t1_pre]
                pre_r = np.sqrt(np.mean(placebo_pre ** 2))
                post_r = np.sqrt(np.mean(gap ** 2))
                ratio = post_r / pre_r if pre_r > 0 else float("inf")
                placebo_ratios.append(ratio)
            except Exception:
                continue

        # --- Rank-based p-value ---
        if len(placebo_ratios) > 0:
            all_ratios = np.array(placebo_ratios + [target_ratio])
            rank = np.sum(all_ratios >= target_ratio)
            p_value = rank / len(all_ratios)
        else:
            p_value = None

        return placebo_gaps, placebo_ratios, target_ratio, p_value

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        target: DonorUnit,
        candidates: List[DonorUnit],
        pre_period: Tuple[int, int],
        post_period: Tuple[int, int],
        covariates: Optional[List[str]] = None,
        min_correlation: float = 0.3,
        max_donors: int = 20,
        n_restarts: int = 10,
    ) -> SyntheticControlResult:
        """
        Run the full synthetic control pipeline end-to-end.

        Parameters
        ----------
        target : DonorUnit
            The treated unit.
        candidates : list of DonorUnit
            All candidate donor units.
        pre_period : tuple of (start, end)
        post_period : tuple of (start, end)
        covariates : list of str, optional
        min_correlation : float
        max_donors : int
        n_restarts : int

        Returns
        -------
        SyntheticControlResult
            Complete results including weights, counterfactual, gap, and
            placebo test statistics.
        """
        # Step 1: Build donor pool
        donor_pool = self.build_donor_pool(
            target, candidates, pre_period, covariates,
            min_correlation=min_correlation, max_donors=max_donors,
        )

        if len(donor_pool) == 0:
            return SyntheticControlResult(
                target_name=target.name,
                donor_names=[],
                weights=np.array([]),
                pre_period=pre_period,
                post_period=post_period,
                pre_rmse=float("inf"),
                counterfactual=np.array([]),
                observed=target.outcomes[post_period[0]:post_period[1]],
                gap=np.array([]),
                placebo_gaps=[],
                placebo_ratios=[],
                target_ratio=float("inf"),
                convergence=False,
            )

        # Step 2: Compute weights
        weights, obj_val = self.compute_synthetic_weights(
            target, donor_pool, pre_period, n_restarts=n_restarts
        )

        # Step 3: Pre-treatment fit
        pre_rmse = self.rmse_pre_weight(target, weights, donor_pool, pre_period)

        # Step 4: Counterfactual prediction
        donor_mat = np.array([u.outcomes for u in donor_pool])
        counterfactual = self.predict_counterfactual(target, weights, donor_mat, post_period)

        # Step 5: Treatment gap
        gap = self.counterfactual_gap(target, counterfactual, post_period)

        # Step 6: Placebo tests
        placebo_gaps, placebo_ratios, target_ratio, p_value = self.compute_placebo_tests(
            target, donor_pool, pre_period, post_period
        )

        convergence = obj_val < 1e6

        return SyntheticControlResult(
            target_name=target.name,
            donor_names=[u.name for u in donor_pool],
            weights=weights,
            pre_period=pre_period,
            post_period=post_period,
            pre_rmse=pre_rmse,
            counterfactual=counterfactual,
            observed=target.outcomes[post_period[0]:post_period[1]],
            gap=gap,
            placebo_gaps=placebo_gaps,
            placebo_ratios=placebo_ratios,
            target_ratio=target_ratio,
            inference_p_value=p_value,
            convergence=convergence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation between two 1D arrays."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return 0.0
        xm, ym = x[mask], y[mask]
        xc = xm - xm.mean()
        yc = ym - ym.mean()
        denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        if denom < 1e-12:
            return 0.0
        return float((xc * yc).sum() / denom)

    def _covariate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Weighted Euclidean distance between covariate vectors."""
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        if self._lambda_v is not None:
            norm = np.sqrt(np.sum(self._lambda_v * diff ** 2))
        else:
            norm = np.sqrt(np.sum(diff ** 2))
        return float(norm)
