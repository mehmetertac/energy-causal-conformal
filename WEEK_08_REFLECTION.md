# Week 08 reflection — energy-causal-conformal

## What did I build?

I built a two-notebook toolkit for **questions ML alone can't answer** — causal effects and forecast guarantees, not just point predictions.

**Causal side (Notebook 1):**

- DiD toy warmup that recovers a known injected treatment effect
- Heterogeneous synthetic-control pipeline: Abadie simplex weights, bootstrap gap uncertainty, in-space and in-time placebos
- DiD and propensity (IPW + DoWhy) cross-checks on the same panel
- Donor-pool and pre-period sensitivity — how much the estimate moves when donors or the fit window change
- Utility rollout read: translate ATT into peak MW avoided and €/MWh, with an explicit decision rule

**Conformal side (Notebook 2):**

- Week 3-style `QuantileLGBM` (pinball loss at P05–P95)
- MAPIE CQR at 80% and 90% with chronological train → cal → test splits (24 h gap)
- First empirical coverage numbers: raw quantiles under-cover; CQR widens to hit nominal
- **Long rolling backtest:** expanding-window CQR over ~365 days, coverage-over-time chart, breakdowns by month and wind regime, pinball comparison to confirm sharpness wasn't destroyed

Both paths share the same discipline: **report honest uncertainty** — bootstrap CIs and placebos for causal effects; empirical coverage on held-out or rolling backtests for intervals.

---

## What's still fuzzy?

### Exchangeability under drift

Split conformal and CQR promise **marginal** coverage when calibration and test points are exchangeable. Hourly wind and load are not: seasonality, fronts, fleet changes, and NWP model updates break that assumption.

Rolling re-conformalization (what Notebook 2 now does) is a practical patch — it tracks slow drift better than one held-out slice — but it is not a proof of infinite-horizon validity. Storm clusters can still produce hours where conditional coverage fails even when marginal coverage looks fine.

### Donor-pool selection

Notebook 1 has explicit sensitivity to donor count and pre-period length on **simulated** data where we know the answer. On real Pecan Street or Low Carbon London meters:

- Opt-in households are not exchangeable with the donor pool
- Donors can be contaminated (neighbours on similar tariffs, later enrolment)
- Pre-period fit quality (RMSPE) and placebo batteries become the only diagnostics — there is no ground-truth recovery check

I know *how* to stress-test donor choice; I have not yet run SC on real LCL dToU vs flat-rate households.

### CQR vs plain split conformal

**CQR** conformalizes already-asymmetric quantile bands — the natural wrap for a pinball-tuned LightGBM stack.

**Plain split conformal** around P50 adds a symmetric buffer from calibration residuals. It ignores the quantile structure the base model already learned.

I chose CQR because Week 3 already trains separate quantile models. I have not run a full backtest-length A/B against split conformal on P50 ± correction; the preference is principled, not empirically settled in this repo.

---

## How do causal counterfactuals and coverage guarantees connect to decisions utilities actually make?

Utilities and TSOs constantly face two different uncertainty problems:

| Question | Tool in this repo | Decision it informs |
|---|---|---|
| Did the tariff / pilot reduce peak load? | Synthetic control, DiD, IPW | **Rollout:** scale the program, adjust the tariff, or stop |
| Can we trust P10–P90 as an 80% envelope? | CQR + rolling coverage backtest | **Reserve / trading:** how much flex to hold, what imbalance exposure to contract |

Both are **contract-grade uncertainty**:

- A **15% peak-load cut** you cannot defend with placebos, donor stability, and a CI that excludes zero is not rollout-ready — even if the point estimate looks good.
- An **80% band that delivers 65%** is unpriced shortage risk — reserve sized from the label is too small, and traders treating the band as a hard limit are misled.

In practice I have seen:

- **Demand-response teams** ask "did the pilot work?" and get a before/after chart confounded by weather. SC/DiD with placebos is the honest answer path.
- **Forecast / balancing desks** publish P10–P90 from a quantile model tuned on pinball loss, then size regulating reserve from the label without checking empirical coverage. The rolling coverage-over-time chart is the monitoring view those teams need — same spirit as checking placebo p-values before scaling a tariff.

The two notebooks are not separate exercises. A utility that runs a peak-shaving tariff (Notebook 1) and a day-ahead wind desk (Notebook 2) needs **both**: evidence the intervention moved load, and evidence the forecast bands can be contracted against. ML that skips either step leaves real financial exposure on the table.

---

## Next increments (for me)

1. Synthetic control on real Low Carbon London dToU vs flat-rate households
2. Re-run Notebook 2 long backtest on sibling Week 3 parquet when available
3. Optional A/B: CQR vs split conformal around P50 on the same rolling walk
