"""
Example: US-China Tariff Counterfactual Analysis.

This script demonstrates a counterfactual analysis of US tariffs on Chinese
imports (Section 301 tariffs imposed in 2018-2019). It uses simulated data
to show the full pipeline: synthetic control construction, narrative reasoning,
and evidence grading.

Usage:
    python examples/china_tariff.py

Note:
    This example uses simulated data for demonstration. Replace with actual
    trade data for production use. An OpenAI API key is required for the
    LLM-powered narrative component.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counterfact_agent import CounterfactualEngine


def generate_simulated_trade_data(
    n_units: int = 15,
    n_periods: int = 20,
    treatment_time: int = 10,
    seed: int = 42,
) -> str:
    """
    Generate simulated panel data for trade analysis.

    Returns the path to a temporary CSV file.

    Parameters
    ----------
    n_units : int
        Number of units (countries/regions).
    n_periods : int
        Number of time periods.
    treatment_time : int
        Time index when the tariff is imposed.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)

    units = [f"unit_{i}" for i in range(n_units)]
    times = list(range(n_periods))

    records = []
    for unit_idx, unit in enumerate(units):
        # Base characteristics
        base_gdp = rng.uniform(100, 500)
        base_trade = rng.uniform(20, 100)
        gdp_trend = rng.uniform(0.5, 2.0)
        trade_trend = rng.uniform(0.2, 1.5)

        for t in times:
            gdp = base_gdp + gdp_trend * t + rng.normal(0, 5)
            trade_volume = base_trade + trade_trend * t + rng.normal(0, 3)

            # Treatment effect for unit_0 (the "China" analogue) after treatment_time
            if unit == "unit_0" and t >= treatment_time:
                # 25-30% reduction in trade volume due to tariffs
                tariff_effect = rng.uniform(25, 30)
                trade_volume -= tariff_effect

            records.append({
                "unit": unit,
                "time": t,
                "outcome": trade_volume,
                "gdp_growth": gdp / base_gdp - 1 + rng.normal(0, 0.01),
                "unemployment": rng.uniform(3, 8) + rng.normal(0, 0.2),
                "industrial_output": base_gdp * 0.3 + rng.normal(0, 5),
                "exchange_rate": 1.0 + rng.normal(0, 0.05),
            })

    df = pd.DataFrame(records)

    # Save to temp file
    tmp_path = os.path.join(os.path.dirname(__file__), "..", "simulated_trade_data.csv")
    df.to_csv(tmp_path, index=False)
    return tmp_path


def main() -> None:
    """Run the US-China tariff counterfactual example."""

    print("=" * 70)
    print("Counterfact-Agent: US-China Tariff Analysis")
    print("=" * 70)
    print()

    # Generate simulated data
    print("[1/4] Generating simulated trade panel data...")
    data_path = generate_simulated_trade_data()
    print(f"      Data saved to: {data_path}")
    print()

    # Initialize engine
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[WARNING] No OPENAI_API_KEY found. Running quantitative analysis only.")
        print("          Set OPENAI_API_KEY to enable LLM narrative reasoning.")
        print()

    engine = CounterfactualEngine(
        openai_api_key=api_key,
        openai_model="gpt-4o",
        min_donor_correlation=0.3,
        max_donors=10,
    )

    # Define the event
    event_description = (
        "US imposed 25% tariffs on approximately $250 billion worth of Chinese "
        "imports under Section 301 of the Trade Act of 1974, implemented in "
        "multiple waves between July 2018 and September 2019. The tariffs targeted "
        "industrial goods, technology products, and consumer goods."
    )

    # Run analysis
    print("[2/4] Running counterfactual analysis...")
    result = engine.analyze(
        event_description=event_description,
        pre_data=data_path,
        covariates=["gdp_growth", "unemployment", "industrial_output", "exchange_rate"],
        target_name="unit_0",
        event_date="2018-07-06",
        pre_period=(0, 10),
        post_period=(10, 20),
        context={
            "sector": "Manufacturing and technology goods",
            "policy_context": "Section 301 investigation of Chinese intellectual property practices",
            "source": "USITC, Census Bureau trade data, academic literature",
        },
    )

    # Print quantitative results
    print()
    print("[3/4] Quantitative Results (Synthetic Control):")
    print("-" * 50)
    if result.sc_result is not None:
        sc = result.sc_result
        print(f"  Target unit:           {sc.target_name}")
        print(f"  Donor pool size:       {len(sc.donor_names)}")
        print(f"  Pre-treatment RMSE:    {sc.pre_rmse:.4f}")
        print(f"  Convergence:           {sc.convergence}")
        if len(sc.gap) > 0:
            print(f"  Mean treatment gap:    {float(np.mean(sc.gap)):+.4f}")
            print(f"  Cumulative gap:        {float(np.sum(sc.gap)):+.4f}")
            print(f"  Inference p-value:     {sc.inference_p_value or 'N/A'}")
        print()
        print("  Donor weights:")
        for name, w in zip(sc.donor_names, sc.weights):
            if w > 0.01:
                print(f"    {name}: {w:.4f}")
    else:
        print("  Synthetic control not available (insufficient data).")

    # Generate report
    print()
    print("[4/4] Generating reports...")
    full_report = engine.full_report(result)
    brief = engine.executive_brief(result)

    reports_dir = os.path.join(os.path.dirname(__file__), "..")
    report_path = os.path.join(reports_dir, "china_tariff_report.md")
    brief_path = os.path.join(reports_dir, "china_tariff_brief.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(brief)

    print(f"  Full report:  {report_path}")
    print(f"  Brief:        {brief_path}")
    print()
    print("=" * 70)
    print("Analysis complete.")

    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)


if __name__ == "__main__":
    main()
