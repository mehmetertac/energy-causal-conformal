# Handover — energy-causal-conformal (Day 6)

**Repo:** https://github.com/mehmetertac/energy-causal-conformal  
**Last updated:** 2026-09-12  
**Branch:** `main`

### Key commit

```
Day 6: brief GP solar appendix, Notebook 2 five-beat tighten, results/ summary figures
```

---

## Project goal

Build the **"questions ML alone can't answer" toolkit** — **effects and guarantees**, not just predictions.

Two capstone notebooks + one interview appendix:

1. **Synthetic control** — tariff-style intervention on residential load (Pecan Street intent; simulated panel now). **Notebook 1 is the finished causal narrative.**
2. **Conformal prediction** — MAPIE CQR on Week 3-style quantile LightGBM with rolling-origin coverage verification (Notebook 2). **Five-beat narrative + money chart export.**
3. **Bayesian GP solar** — compressed GPyTorch ExactGP demo (Notebook 3). **Interview appendix only; conformal wins in production.**

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
| Notebook 1 ([notebooks/01_synthetic_control.ipynb](notebooks/01_synthetic_control.ipynb)) — summary export | Done |
| Conformal modules ([src/conformal/](src/conformal/)) — synthetic wind, QuantileLGBM, CQR, coverage, rolling backtest | Done |
| GP solar module ([src/conformal/gp_solar.py](src/conformal/gp_solar.py)) — not exported from package `__init__` | Done |
| Notebook 2 ([notebooks/02_conformal_forecast.ipynb](notebooks/02_conformal_forecast.ipynb)) — five-beat tighten, crisp coverage table | Done |
| Notebook 3 ([notebooks/03_gp_solar.ipynb](notebooks/03_gp_solar.ipynb)) — RBF + periodic ExactGP, vs CQR paragraph | Done |
| Shared plotting ([src/plotting.py](src/plotting.py)) — style + `results/` export | Done |
| [WEEK_08_REFLECTION.md](WEEK_08_REFLECTION.md) | Done |
| LCL loader + Pecan Street stub | Done |
| Unit tests (causal + conformal + gp_solar + data) | Done |
| Pre-commit hooks (file size + pytest) | Done |

---

## Data situation

| Source | Role | Status |
|---|---|---|
| **Pecan Street Dataport** | Intended for Notebook 1 | Stub only |
| **Low Carbon London** | Open substitute (dToU vs flat 2013) | Loader + manual zip download |
| **Simulated panel** | DiD toy + SC experiment | Used in tests + Notebook 1 |
| **Synthetic wind** | Conformal experiments | [src/conformal/simulate.py](src/conformal/simulate.py); optional sibling Week 3 parquet |
| **Synthetic solar** | GP demo (Notebook 3) | [src/conformal/gp_solar.py](src/conformal/gp_solar.py) |

See [data/README.md](data/README.md). CI uses fixtures only — no network.

---

## Repo layout

```
energy-causal-conformal/
├── README.md
├── AGENT.md
├── handover.md
├── WEEK_08_REFLECTION.md
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
├── notebooks/03_gp_solar.ipynb
├── src/causal/
├── src/conformal/
│   ├── backtest.py
│   ├── coverage.py
│   ├── cqr.py
│   ├── gp_solar.py          # not in __init__ (GPyTorch optional)
│   ├── quantile_lgbm.py
│   ├── simulate.py
│   └── split.py
├── src/plotting.py
├── src/data/
├── tests/
└── results/                 # summary PNGs (gitignored)
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
jupyter notebook notebooks/03_gp_solar.ipynb
```

Summary figures (on notebook run): `results/nb01_treated_vs_synthetic.png`, `results/nb02_coverage_over_time.png`, `results/nb03_gp_solar_bands.png`.

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
| `WindSimulationConfig` | Synthetic wind series config (`seasonal_heteroskedasticity`, `drift_day`) |
| `simulate_wind_forecast()` | Hourly DE-style wind with heteroskedastic tails |
| `feature_columns()` | Modeling feature names for synthetic wind |
| `QuantileLGBM` | Week 3 port — one LightGBM per quantile (P05–P95) |
| `WEEK3_MODEL_PARAMS` | Locked hyperparams from Week 3 Optuna run |
| `chronological_conformal_split(...)` | Train / cal / test with optional gap |
| `run_conformal_cqr(...)` | Fit quantile LGBM + MAPIE CQR at 80%/90% |
| `ConformalForecastResult` | Raw vs CQR coverage on test block |
| `RollingOriginConfig` | Hour-based expanding-window backtest settings |
| `rolling_origin_backtest(...)` | Expanding train + fixed cal/test CQR walk |
| `RollingBacktestResult` | `fold_table`, `predictions` with month/regime tags |
| `coverage_by_month(...)` / `coverage_by_regime(...)` | Grouped coverage + width |
| `pinball_comparison(...)` | Raw vs CQR pinball at P10/P50/P90 |
| `pi_coverage(...)` / `evaluate_intervals(...)` / `pinball_loss(...)` | Metrics |
| `CoverageResult` | `coverage`, `nominal`, `coverage_gap`, `mean_width` |

### GP solar (isolated) — `src/conformal/gp_solar.py`

Not exported from `src.conformal.__init__` — conformal tests stay GPyTorch-free.

| Symbol | Purpose |
|---|---|
| `SolarSimulationConfig` | Short-horizon PV series config |
| `simulate_solar_output()` | Hourly solar with diurnal envelope |
| `fit_gp_solar(train, test, n_iter=40)` | ExactGP RBF + periodic; returns mean/std |
| `GPSolarForecast` | Posterior mean, std, train/test arrays |

### Plotting — `src/plotting.py`

| Symbol | Purpose |
|---|---|
| `setup_style()` | Shared matplotlib defaults for notebooks |
| `save_summary_figure(fig, stem)` | Write `results/{stem}.png` |
| `bootstrap_notebook_paths()` | Add repo root to `sys.path` from `notebooks/` cwd |

### Data — `src/data/`

| Symbol | Purpose |
|---|---|
| `load_lcl_sample()` | Load committed LCL fixture |
| `load_lcl_csv(path)` | Parse LCL CSV |
| `load_pecan_street()` | Stub — raises until Dataport files exist |

---

## Suggested next step (Day 7+)

1. **Synthetic control on LCL** — real dToU vs flat-rate households (no injected effect)
2. **Optional:** wire sibling `wind-quantile-forecast` parquet as default Notebook 2 input
3. **Optional A/B:** CQR vs split conformal around P50 on the same rolling walk

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/energy-causal-conformal.git`
- Python venv at `.venv/` (gitignored)
- `mapie>=1.0` for `ConformalizedQuantileRegressor` (MAPIE v1 API)
- `torch>=2.0` + `gpytorch>=1.11` for Notebook 3; GP tests import module directly (skip if GPyTorch absent)
