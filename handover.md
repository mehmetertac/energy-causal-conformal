# Handover — energy-causal-conformal (Day 4)

**Repo:** https://github.com/mehmetertac/energy-causal-conformal  
**Last updated:** 2026-09-11  
**Branch:** `main`

### Key commit

```
8be442b Day 4: MAPIE CQR on Week 3 quantile LightGBM, Notebook 2
```

---

## Project goal

Build the **"questions ML alone can't answer" toolkit** — **effects and guarantees**, not just predictions.

Two capstone notebooks:

1. **Synthetic control** — tariff-style intervention on residential load (Pecan Street intent; simulated panel now). **Notebook 1 is the finished causal narrative.**
2. **Conformal prediction** — MAPIE CQR on Week 3-style quantile LightGBM with first empirical coverage numbers (Notebook 2). Long rolling backtest is next.

Agent workflow rules: [AGENT.md](AGENT.md)

---

## What is done

| Item | Status |
|---|---|
| Repo scaffold (`src/`, `notebooks/`, `data/`, `results/`, `docs/`) | Done |
| [AGENT.md](AGENT.md) + [handover.md](handover.md) | Done |
| Causal mental model ([docs/causal_mental_model.md](docs/causal_mental_model.md)) | Done |
| Conformal mental model ([docs/conformal_mental_model.md](docs/conformal_mental_model.md)) | Done |
| DiD simulator + estimator | Done |
| Heterogeneous SC simulator + estimator + inference/placebos | Done |
| Propensity IPW + DoWhy cross-check | Done |
| Warmup notebook ([notebooks/00_did_toy_warmup.ipynb](notebooks/00_did_toy_warmup.ipynb)) | Done |
| Notebook 1 ([notebooks/01_synthetic_control.ipynb](notebooks/01_synthetic_control.ipynb)) | Done |
| Conformal modules ([src/conformal/](src/conformal/)) — synthetic wind, QuantileLGBM, CQR, coverage | Done |
| Notebook 2 ([notebooks/02_conformal_forecast.ipynb](notebooks/02_conformal_forecast.ipynb)) — raw vs CQR coverage | Done |
| LCL loader + Pecan Street stub | Done |
| Unit tests (causal + conformal + data) | Done |
| Pre-commit hooks (file size + pytest) | Done |

---

## Data situation

| Source | Role | Status |
|---|---|---|
| **Pecan Street Dataport** | Intended for Notebook 1 | Stub only |
| **Low Carbon London** | Open substitute (dToU vs flat 2013) | Loader + manual zip download |
| **Simulated panel** | DiD toy + SC experiment | Used in tests + Notebook 1 |
| **Synthetic wind** | Conformal experiments | [src/conformal/simulate.py](src/conformal/simulate.py); optional sibling Week 3 parquet |

See [data/README.md](data/README.md). CI uses fixtures only — no network.

---

## Repo layout

```
energy-causal-conformal/
├── README.md
├── AGENT.md
├── handover.md
├── requirements.txt
├── pytest.ini
├── .pre-commit-config.yaml
├── scripts/check_file_size.py
├── docs/causal_mental_model.md
├── docs/conformal_mental_model.md
├── data/README.md
├── notebooks/00_did_toy_warmup.ipynb
├── notebooks/01_synthetic_control.ipynb
├── notebooks/02_conformal_forecast.ipynb
├── src/causal/
├── src/conformal/
├── src/data/
├── tests/
└── results/
```

---

## How to run

```powershell
cd energy-causal-conformal
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pre-commit install

pytest tests/ -q
jupyter notebook notebooks/00_did_toy_warmup.ipynb
jupyter notebook notebooks/01_synthetic_control.ipynb
jupyter notebook notebooks/02_conformal_forecast.ipynb
```

---

## Core module API

### Simulator — `src/causal/simulate.py`

| Symbol | Purpose |
|---|---|
| `SimulationConfig` | Household counts, days, ATT (kW), weather jump, peak hours (DiD toy) |
| `simulate_residential_load()` | Hourly panel with treated/control, post flag, confounder |
| `SynthSimulationConfig` | Heterogeneous panel config for synthetic control |
| `simulate_heterogeneous_load()` | Pecan-style IDs, household-specific load shapes |
| `summarize_panel()` | Row/household counts for sanity checks |

### DiD — `src/causal/did.py`

| Symbol | Purpose |
|---|---|
| `estimate_did(frame, peak_hours=...)` | Two-way OLS; returns `DidResult` |
| `naive_before_after(frame, peak_hours=...)` | Treated-only before/after |
| `DidResult` | `att_kw`, `std_error`, `ci_low`, `ci_high`, `n_obs` |

### Synthetic control — `src/causal/synth.py`

| Symbol | Purpose |
|---|---|
| `inject_peak_reduction(...)` | Known % peak-load cut post-date |
| `daily_peak_series(...)` | Daily mean peak-hour kW |
| `fit_synthetic_control(...)` | Abadie simplex weights |
| `gap_uncertainty(...)` | Bootstrap CI on post-period mean gap |
| `in_space_placebos(...)` / `in_time_placebos(...)` | Placebo p-values |
| `SyntheticControlResult` | `weights`, `treated`, `synthetic`, `att_kw`, `att_pct` |

### Propensity — `src/causal/propensity.py`

| Symbol | Purpose |
|---|---|
| `estimate_ipw_att(...)` | Logistic IPW ATT |
| `estimate_dowhy_att(...)` | DoWhy PS-weighting |
| `PropensityResult` | `att_kw`, `n_treated`, `n_control`, `method` |

### Conformal — `src/conformal/`

| Symbol | Purpose |
|---|---|
| `WindSimulationConfig` | Synthetic wind series config |
| `simulate_wind_forecast()` | Hourly DE-style wind with heteroskedastic tails |
| `feature_columns()` | Modeling feature names for synthetic wind |
| `QuantileLGBM` | Week 3 port — one LightGBM per quantile (P05–P95) |
| `WEEK3_MODEL_PARAMS` | Locked hyperparams from Week 3 Optuna run |
| `chronological_conformal_split(...)` | Train / cal / test with optional gap |
| `run_conformal_cqr(...)` | Fit quantile LGBM + MAPIE CQR at 80%/90% |
| `ConformalForecastResult` | Raw vs CQR coverage on test block |
| `pi_coverage(...)` / `evaluate_intervals(...)` | Empirical coverage + width metrics |
| `CoverageResult` | `coverage`, `nominal`, `coverage_gap`, `mean_width` |

### Data — `src/data/`

| Symbol | Purpose |
|---|---|
| `load_lcl_sample()` | Load committed LCL fixture |
| `load_lcl_csv(path)` | Parse LCL CSV |
| `load_pecan_street()` | Stub — raises until Dataport files exist |

---

## Suggested next step (Day 5+)

1. **Long rolling conformal backtest** — refit/re-conformalize on expanding windows; coverage by regime
2. **Synthetic control on LCL** — real dToU vs flat-rate households (no injected effect)
3. **Optional:** wire sibling `wind-quantile-forecast` parquet as default Notebook 2 input

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/energy-causal-conformal.git`
- Python venv at `.venv/` (gitignored)
- `mapie>=1.0` for `ConformalizedQuantileRegressor` (MAPIE v1 API)
- GPyTorch in requirements for later GP solar example; conformal tests do not import it
