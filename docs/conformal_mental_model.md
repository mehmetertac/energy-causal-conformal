# Conformal mental model for forecast intervals

This note maps conformal prediction ideas to day-ahead wind / load forecasting. Notebook 2 ([`notebooks/02_conformal_forecast.ipynb`](../notebooks/02_conformal_forecast.ipynb)) applies them to the Week 3 quantile LightGBM stack.

---

## What split conformal gives you

Split (inductive) conformal prediction:

1. **Fit** a base model on a training set.
2. **Score** nonconformity on a held-out **calibration** set — how wrong was the model?
3. Take a **quantile** of those scores and add it to future predictions to form intervals.

The guarantee: under **exchangeability** of calibration and test points, intervals have **marginal coverage** at the nominal level (e.g. 90%) with **no distributional assumption** on the errors.

**Energy example:** A day-ahead wind desk publishes P10–P90 bands. Split conformal asks: "If I widen those bands by the 90th percentile of past misses, do I cover 90% of future hours?" — and proves it, if the fine print holds.

---

## CQR — conformalized quantile regression

Week 3 trains separate LightGBM models at P10, P50, and P90 with pinball loss. **CQR** (Romano et al., 2019) is the natural wrap:

- Nonconformity score for observation `(X, y)` with lower bound `q_lo` and upper bound `q_hi`:

  `score = max(q_lo − y, y − q_hi)`

- Calibrate a correction `Q` as the `(1 − α)(1 + 1/n)` quantile of scores on the calibration set.
- Conformalized interval: `[q_lo − Q, q_hi + Q]`.

CQR **widens** under-covering quantile bands until empirical coverage matches nominal — without retraining the base quantile models.

**Energy example:** Week 3 default LightGBM hit ~59% P10–P90 coverage (nominal 80%). CQR adds a uniform buffer from calibration residuals so dispatchers can trust the label again.

---

## Exchangeability — the fine print

Marginal coverage guarantees require calibration and test points to be **exchangeable** (roughly: identically distributed, order irrelevant).

Hourly wind and load are **not** exchangeable:

- Strong **autocorrelation** and seasonality
- **Regime shifts** (fronts, calm spells, holidays)
- **Concept drift** as NWP models or fleets change

**Do not** shuffle time series or use MAPIE's random `train_conformalize_test_split` for production wind/load backtests.

Instead:

| Strategy | Idea |
|---|---|
| **Blocked / chronological split** | Train → calibrate → test in time order; optional gap (24 h) between blocks |
| **Rolling calibration** | Refit or re-conformalize on an expanding past window — **implemented** in [`src/conformal/backtest.py`](../src/conformal/backtest.py) |
| **EnbPI / ACI** | MAPIE time-series methods that update scores or adapt quantiles online |

This repo now has both a **single chronological split** (honest first numbers) and a **long rolling backtest** (expanding train, fixed cal/test blocks, 30-day step). Notebook 2's **coverage-over-time chart** is the production monitoring view — does empirical coverage track nominal month after month?

---

## Raw quantiles vs conformalized intervals

| | Raw quantile model (Week 3) | CQR (MAPIE) |
|---|---|---|
| **Training objective** | Pinball loss at each quantile | Same base models |
| **Calibration** | None (or manual isotonic) | Conformal correction from held-out scores |
| **Coverage guarantee** | None — often under-covers | Marginal, if exchangeability holds |
| **Typical finding** | P10–P90 covers ~55–65% | Hits ~80% after widening |
| **Interval width** | Narrower (optimistic) | Wider (honest) |

**Dispatch read:** Under-calibrated P10–P90 bands shrink the reserve buffer on paper. Conformal correction buys **trustworthy labels** at the cost of **wider** intervals — the right trade for balancing and imbalance risk.

---

## What to report in production

- **Empirical coverage** on a long held-out backtest — nominal 90% is not enough on its own.
- **Coverage over time** — rolling-origin fold table plotted by test month (Notebook 2 money chart).
- **Mean interval width** alongside coverage (efficiency vs guarantee).
- **Coverage by regime** (high-wind vs low-wind terciles) and by calendar month.
- **Pinball loss** at P10/P50/P90 — confirm CQR widened for coverage, not to destroy median sharpness.
- **Calibration window** length and refresh policy.

See also: [`docs/causal_mental_model.md`](causal_mental_model.md) — honest uncertainty on causal effects uses the same discipline.

---

## Further reading in this repo

| Doc | Purpose |
|---|---|
| [README.md](../README.md) | Project goal, two-notebook arc |
| [handover.md](../handover.md) | Module API and status |
| [AGENT.md](../AGENT.md) | Agent workflow rules |
