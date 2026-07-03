"""
Example: EU Carbon Tax Counterfactual Analysis.

This script demonstrates a counterfactual analysis of the EU Emissions Trading
System (EU ETS) on carbon emissions. It uses simulated data to show what
emissions would look like without carbon pricing.

Usage:
    python examples/carbon_tax.py

Note:
    This example uses simulated data for demonstration. An OpenAI API key
    is required for the LLM-powered narrative component.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counterfact_agent import CounterfactualEngine


def generate_simulated_emissions_data(
    n_units: int = 20,
    n_periods: int = 30,
    treatment_time: int = 15,
    seed: int = 123,
) -> str:
    """
    Generate simulated panel data for emissions analysis.

    Parameters
    ----------
    n_units : int
        Number of units (countries).
    n_periods : int
        Number of time periods (years).
    treatment_time : int
        Time index when carbon pricing was introduced.
    seed : int
        Random seed.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)

    units = [f"country_{i}" for i in range(n_units)]
    times = list(range(n_periods))

    records = []
    for unit_idx, unit in enumerate(units):
        # Base with some heterogeneity
        base_emissions = rng.uniform(50, 200)
        gdp = rng.uniform(200, 1000)
        renewable_share = rng.uniform(0.05, 0.30)
        energy_intensity = rng.uniform(0.1, 0.5)

        for t in times:
            # Baseline emissions trend (slight upward)
            trend = rng.uniform(0.3, 1.0) * t
            emissions = base_emissions + trend + rng.normal(0, 3)

            # Treatment: EU ETS target (country_0) after treatment_time
            if unit == "country_0" and t >= treatment_time:
                # ~15-20% emission reduction due to carbon pricing
                reduction = rng.uniform(15, 20) + 0.5 * (t - treatment_time)
                emissions -= reduction

            records.append({
                "unit": unit,
                "time": t,
                "outcome": max(emissions, 10),
                "gdp_per_capita": gdp + rng.normal(0, 10),
                "industrial_share": rng.uniform(0.15, 0.35) + rng.normal(0, 0.005),
                "renewable_share": renewable_share + 0.005 * t + rng.normal(0, 0.005),
                "energy_intensity": energy_intensity - 0.003 * t + rng.normal(0, 0.002),
                "population": rng.uniform(5, 80) + rng.normal(0, 0.5),
                "r_and_d_spending": rng.uniform(1.0, 4.0) + 0.05 * t + rng.normal(0, 0.1),
            })

    df = pd.DataFrame(records)

    tmp_path = os.path.join(os.path.dirname(__file__), "..", "simulated_emissions_data.csv")
    df.to_csv(tmp_path, index=False)
    return tmp_path


def main() -> None:
    """Run the EU carbon tax counterfactual example."""

    print("=" * 70)
    print("Counterfact-Agent: EU Carbon Pricing Analysis")
    print("=" * 70)
    print()

    # Generate data
    print("[1/4] Generating simulated emissions panel data...")
    data_path = generate_simulated_emissions_data()
    print(f"      Data saved to: {data_path}")
    print()

    # Initialize engine
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[WARNING] No OPENAI_API_KEY found. Running quantitative analysis only.")
        print()

    engine = CounterfactualEngine(
        openai_api_key=api_key,
        openai_model="gpt-4o",
        min_donor_correlation=0.3,
        max_donors=15,
    )

    # Define the event
    event_description = (
        "The European Union Emissions Trading System (EU ETS) was launched in 2005 "
        "as the world's first major carbon market. Phase 3 (2013-2020) introduced "
        "auctioning as the default allocation method and expanded to cover ~45% of "
        "EU greenhouse gas emissions. The carbon price averaged approximately 25 EUR "
        "per tonne CO2 during this period."
    )

    # Run analysis
    print("[2/4] Running counterfactual analysis...")
    result = engine.analyze(
        event_description=event_description,
        pre_data=data_path,
        covariates=[
            "gdp_per_capita", "industrial_share", "renewable_share",
            "energy_intensity", "r_and_d_spending",
        ],
        target_name="country_0",
        event_date="2013-01-01",
        pre_period=(0, 15),
        post_period=(15, 30),
        context={
            "sector": "Energy and industrial emissions",
            "policy_context": "EU ETS Phase 3 with auctioning, covering power, industry, aviation",
            "source": "European Environment Agency, World Bank Carbon Pricing Dashboard",
        },
    )

    # Print results
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
        if len(sc.donor_names) > 0:
            print()
            print("  Top donors by weight:")
            sorted_donors = sorted(
                zip(sc.donor_names, sc.weights),
                key=lambda x: x[1], reverse=True
            )
            for name, w in sorted_donors[:5]:
                if w > 0.01:
                    print(f"    {name}: {w:.4f}")
    else:
        print("  Synthetic control not available.")

    # Generate report
    print()
    print("[4/4] Generating reports...")
    full_report = engine.full_report(result)
    brief = engine.executive_brief(result)

    reports_dir = os.path.join(os.path.dirname(__file__), "..")
    report_path = os.path.join(reports_dir, "carbon_tax_report.md")
    brief_path = os.path.join(reports_dir, "carbon_tax_brief.md")

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
