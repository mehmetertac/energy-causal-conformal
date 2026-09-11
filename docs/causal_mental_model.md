# Causal mental model for energy interventions

This note maps core causal-inference ideas to utility and retail-energy questions. The Day 1 DiD toy in [`notebooks/00_did_toy_warmup.ipynb`](../notebooks/00_did_toy_warmup.ipynb) and Day 2 synthetic control in [`notebooks/01_synthetic_control.ipynb`](../notebooks/01_synthetic_control.ipynb) make each concept concrete.

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

Synthetic control builds a weighted combination of control regions to approximate that path. See [`notebooks/01_synthetic_control.ipynb`](../notebooks/01_synthetic_control.ipynb).

---

## Synthetic control

When you have **one treated unit** (a feeder, city, or aggregated pilot group) and many untreated donors, synthetic control (SC) constructs a counterfactual as a **convex combination of donors**:

`Y_synthetic,t = Σ_j w_j · Y_donor_j,t` with `w_j ≥ 0` and `Σ w_j = 1`

Weights are fit on the **pre-intervention** window so the synthetic path tracks the treated unit closely before the tariff. After rollout, the same weights define **Y(0)** — what treated load would have been without the policy.

**Energy example:** A Pecan Street pilot enrols 5 high-usage Austin homes in a peak-shaving tariff. Donors are 30 similar untreated homes. SC finds weights (e.g. 0.4×Home_A + 0.35×Home_B + …) that match the pilot group's pre-tariff evening peak. The post-tariff gap between actual and synthetic peak load is the effect estimate.

### Why SC beats naive before/after for a single treated unit

Naive before/after compares the treated unit to **itself** across time. Any shared shock (cold snap, bill increase, EV adoption) moves the treated series and is mistaken for policy impact.

SC subtracts a donor combination that experienced the **same common shocks** but did not receive the tariff. If donors and treated tracked in the pre-period, the synthetic path carries forward the confounder; only the policy-specific divergence remains.

DiD needs two large groups with parallel trends. SC is the right tool when treatment applies to **one aggregate unit** and you need a bespoke counterfactual, not a simple control mean.

### Placebo tests

Placebos ask: "Could a fake treatment produce a gap this large?"

| Test | Idea | Pass criterion |
|---|---|---|
| **In-space** | Pretend each donor was treated; refit SC and measure post gaps | Real treated gap should be extreme vs placebo distribution |
| **In-time** | Pick a fake pre-period date as "treatment" | No significant gap should appear before the true rollout |

Placebos do not replace uncertainty intervals, but they sanity-check whether the estimated effect is distinguishable from noise.

Notebook 1 reports both: a **bootstrap interval** on the post-period mean gap (synthetic path held fixed) and **in-space + in-time placebos**. On the simulator the SC ATT recovers the injected 15% peak cut; that recovery check is impossible on a real tariff.

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

Notebook 1 cross-checks SC with two-way DiD OLS and a propensity-weighted ATT (sklearn IPW and DoWhy `backdoor.propensity_score_weighting`) on household peak-load *changes*. In the simulator assignment is as-if random, so IPW is a method sketch for the opt-in case — high-usage homes self-selecting into a tariff.

---

## Honest uncertainty

- Report **confidence intervals** on causal effects, not only point estimates.
- For synthetic control, pair the gap interval with placebo p-values; the interval does not capture weight-estimation uncertainty.
- For conformal prediction (Notebook 2), verify **empirical coverage** on a long rolling backtest — nominal 90% is not enough on its own. See [`docs/conformal_mental_model.md`](conformal_mental_model.md).

---

## Further reading in this repo

| Doc | Purpose |
|---|---|
| [README.md](../README.md) | Project goal, setup, tariff narrative |
| [handover.md](../handover.md) | Current status and module API |
| [docs/conformal_mental_model.md](conformal_mental_model.md) | Split conformal, CQR, exchangeability caveats |
| [data/README.md](../data/README.md) | Pecan Street intent + LCL substitute |
| [AGENT.md](../AGENT.md) | Agent workflow rules |
