# Two questions your forecast can't answer

Most ML in energy stops at prediction: next-hour load, day-ahead wind, tomorrow's price. That is necessary. It is not sufficient for the decisions that actually move money.

Utilities and balancing desks constantly face two different uncertainty problems — and conflating them is how pilots get rolled out on confounded evidence, and how reserve gets sized from bands that do not cover.

## 1. What changed because of the intervention?

You ran a peak-shaving tariff on five homes. Load dropped 12% in the post period. **Did the tariff work?**

A before/after chart cannot answer that. Weather moved. Neighbours changed behaviour. The treated homes were not randomly assigned. What you need is a **counterfactual**: what would evening peak have been without the tariff?

Synthetic control builds that counterfactual from untreated donors, estimates an average treatment effect, and stress-tests with placebos on fake treated units and fake intervention dates. On simulated data where we inject a known **15% peak cut**, SC recovers **−15.0%**; naive before/after does not.

That is rollout evidence — not a forecast accuracy metric.

## 2. How confident are you entitled to be?

Your quantile model publishes P10–P90 and labels it an **80% envelope**. Traders and reserve planners treat that label as contract-grade. **Does it cover 80% of held-out hours?**

On a 365-day rolling backtest of synthetic day-ahead wind, raw P10–P90 delivers **~58%** empirical coverage (nominal 80%). Conformalized quantile regression (CQR) widens the bands and brings rolling mean coverage to **~76%**. You pay in width (~6,400 MW vs ~4,000 MW); you buy a label you can monitor month after month.

Bayesian GP intervals answer a different question — *what does the model believe?* — and scale poorly to production row counts. Conformal intervals answer *does the label match held-out frequency?* That is what a reserve desk needs.

## Same discipline, two notebooks

| Question | Tool | Decision it informs |
|---|---|---|
| Did the pilot work? | Synthetic control + placebos | Scale the tariff, adjust it, or stop |
| Can we trust the band? | CQR + coverage-over-time backtest | How much flex to hold, what imbalance to contract |

Both are **contract-grade uncertainty**. ML that skips either step leaves real financial exposure on the table.

**Repo:** https://github.com/mehmetertac/energy-causal-conformal
