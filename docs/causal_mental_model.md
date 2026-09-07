# Causal mental model for energy interventions

This note maps core causal-inference ideas to utility and retail-energy questions. The Day 1 DiD toy in [`notebooks/00_did_toy_warmup.ipynb`](../notebooks/00_did_toy_warmup.ipynb) makes each concept concrete.

---

## Potential outcomes

For household *i* at time *t*, define:

- **Y(1)**: load if the new tariff applies
- **Y(0)**: load if the old tariff applies

We only observe one outcome per household-period. The other is the **counterfactual**.

**Energy example:** Did the dynamic time-of-use tariff reduce evening peak kWh for treated homes?

---

## Counterfactuals

The counterfactual is the untreated path we cannot directly observe for treated homes after rollout.

**Energy example:** What would Austin residential load have looked like in July 2023 without the EV pilot incentive?

Synthetic control (later this week) builds a weighted combination of control regions to approximate that path.

---

## Why naive before/after proves nothing

Comparing treated load before vs after the tariff confounds the policy with everything else that moved:

- Weather (cold snap → heating load up)
- Seasonality (summer AC)
- Macro / bill shock
- Technology adoption (heat pumps, EVs)

**Energy example:** "Load dropped 8% after the tariff" is not causal evidence if control homes also dropped 6% because of a mild week.

---

## Confounders

Confounders influence both treatment assignment and the outcome.

| Confounder | Why it matters |
|---|---|
| Temperature | Drives heating/cooling load |
| Occupancy / WFH | Shifts midday demand |
| PV / EV ownership | Changes net load shape |
| Income / dwelling type | Correlates with tariff uptake |

**Energy example:** High-usage homes may self-select into time-of-use pilots, biasing a simple treated-vs-control comparison.

---

## Parallel trends

Difference-in-differences assumes treated and control groups would have followed **parallel paths** absent treatment, after adjusting for group fixed effects.

Formally, for pre-period windows:  
`E[Y_treated,t - Y_control,t | no treatment] ≈ constant` over time.

**Energy example:** If control and treated London homes tracked each other through 2012, DiD can attribute post-2013 divergence to the dynamic tariff — if the shock is common (weather) it cancels in the difference.

---

## Where propensity scoring fits

When treatment is **not** as-if random (pilots, opt-in tariffs), match or weight units by estimated treatment probability `P(T=1 | X)` using pre-period covariates (AC propensity, pre-load, dwelling type).

**Energy example:** Match EV-trial enrollees to similar non-enrollees on pre-period evening load and feeder capacity before estimating shift effects.

DoWhy / EconML enter later in the week; Day 1 uses plain two-way DiD OLS to calibrate intuition.

---

## Honest uncertainty

- Report **confidence intervals** on causal effects, not only point estimates.
- For conformal prediction (Notebook 2), verify **empirical coverage** on a long rolling backtest — nominal 90% is not enough on its own.

---

## Further reading in this repo

| Doc | Purpose |
|---|---|
| [README.md](../README.md) | Project goal, setup, tariff narrative |
| [handover.md](../handover.md) | Current status and module API |
| [data/README.md](../data/README.md) | Pecan Street intent + LCL substitute |
| [AGENT.md](../AGENT.md) | Agent workflow rules |
