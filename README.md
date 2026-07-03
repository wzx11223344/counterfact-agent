# Counterfact-Agent

**Policy Counterfactual Reasoning Agent** -- Given a policy event, constructs structured counterfactual scenarios using both quantitative methods and LLM-powered narrative reasoning.

## Bold Claim

Unlike generic "ask ChatGPT" tools, this agent **actually runs synthetic control methodology** (Abadie et al., 2010) where data is available, falling back to structured LLM reasoning only when quantitative data is insufficient. Every output distinguishes between **quantitative bounds** and **qualitative direction**.

## Architecture

```
counterfact-agent/
├── README.md
├── requirements.txt
├── counterfact_agent/
│   ├── __init__.py
│   ├── engine.py             -- Main counterfactual pipeline
│   ├── synthetic_control.py   -- Pure Python SC method (Abadie et al. 2010)
│   ├── narrative.py           -- LLM-powered mechanism narratives
│   ├── evidence.py            -- Evidence grading and sourcing
│   └── reporter.py            -- Structured report output
├── config/
│   └── scenarios.yaml         -- Pre-built scenario templates
├── examples/
│   ├── china_tariff.py        -- US-China tariff counterfactual
│   └── carbon_tax.py          -- EU carbon tax counterfactual
└── tests/
    └── test_counterfact.py
```

## How It Works

### 1. Quantitative Layer: Synthetic Control Method

The agent implements a pure Python synthetic control method (no LLM needed):

- **Donor Pool Construction**: Selects comparable units that did not receive the treatment
- **Constrained Optimization**: Minimizes `||X1 - X0 * W||` subject to weights >= 0 and sum(W) = 1
- **Counterfactual Prediction**: Uses learned weights to predict what would have happened
- **Placebo Tests**: In-space placebo to assess significance
- **Treatment Effect Estimation**: Gap between observed and synthetic counterfactual

Based on: Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies.

### 2. Qualitative Layer: LLM Narrative Reasoning

When quantitative data is insufficient or to complement the SC results:

- **Mechanism Narratives**: Explains WHY the gap exists (direct effects, spillovers, equilibrium adjustments)
- **Confidence Discussion**: Explicitly grades uncertainty with tags
- **Policy Implications**: Translates findings into actionable insights

Every LLM output includes confidence tags: `[HIGH_CONF]` `[MED_CONF]` `[LOW_CONF]` `[SPECULATIVE]`

### 3. Evidence Grading

All claims are graded A/B/C/D based on:
- Verifiability
- Source credibility
- Methodological rigor
- Corroboration across independent sources

### 4. Report Output

- Full markdown reports
- 2-page executive briefs for policymakers
- Sensitivity analysis
- Interactive HTML with toggleable sections

## Quick Start

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from counterfact_agent import CounterfactualEngine

engine = CounterfactualEngine(openai_api_key="...")

result = engine.analyze(
    event_description="US imposed 25% tariff on Chinese steel imports in March 2018",
    pre_data="path/to/pre_treatment.csv",
    covariates=["gdp_growth", "unemployment", "industrial_output"]
)

print(result.executive_brief())
```

### Example Scripts

```bash
# US-China tariff counterfactual analysis
python examples/china_tariff.py

# EU carbon tax counterfactual analysis
python examples/carbon_tax.py
```

## Key Differentiators

| Feature | Counterfact-Agent | Generic LLM Tools |
|---------|------------------|-------------------|
| Synthetic Control Math | Runs actual optimization | Cannot do math |
| Donor Pool Selection | Algorithmic | Ad-hoc at best |
| Placebo Testing | Quantitative significance | None |
| Evidence Grading | Structured A/B/C/D | No systematic grading |
| Confidence Tags | Per-claim explicit | Vague hedging |
| Sensitivity Analysis | Systematic parameter sweep | Not available |
| Reproducibility | Deterministic SC + fixed LLM temp | Non-deterministic |

## Scenarios (Pre-built Templates)

- **Trade Policy**: Tariff imposition / removal
- **Labor Market**: Minimum wage changes
- **Environmental**: Carbon pricing impact
- **Education**: School funding reform
- **Health**: Public health interventions

See `config/scenarios.yaml` for full template configurations.

## Requirements

- Python >= 3.10
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- openai >= 1.0.0
- pyyaml >= 6.0

## License

MIT
