# Handover — energy-causal-conformal (Day 1)

**Repo:** https://github.com/mehmetertac/energy-causal-conformal  
**Last updated:** 2026-09-07  
**Branch:** `main`

### Key commit

```
49ea1ea Day 1: causal mental model, DiD toy, repo scaffold, LCL data loader
```

---

## Project goal

Build the **"questions ML alone can't answer" toolkit** — **effects and guarantees**, not just predictions.

Two capstone notebooks (later this week):

1. **Synthetic control** — tariff-style intervention on residential load (Pecan Street intent; LCL substitute now)
2. **Conformal prediction** — MAPIE intervals on a Week 3-style LightGBM load forecast with long rolling backtest and empirical coverage checks

Agent workflow rules: [AGENT.md](AGENT.md)

---

## What is done (Day 1)

| Item | Status |
|---|---|
| Repo scaffold (`src/`, `notebooks/`, `data/`, `results/`, `docs/`) | Done |
| [AGENT.md](AGENT.md) + [handover.md](handover.md) | Done |
| Causal mental model ([docs/causal_mental_model.md](docs/causal_mental_model.md)) | Done |
| DiD simulator ([src/causal/simulate.py](src/causal/simulate.py)) | Done |
| DiD estimator ([src/causal/did.py](src/causal/did.py)) | Done |
| Warmup notebook ([notebooks/00_did_toy_warmup.ipynb](notebooks/00_did_toy_warmup.ipynb)) | Done |
| LCL loader + download helpers ([src/data/london_smartmeter.py](src/data/london_smartmeter.py)) | Done |
| Pecan Street stub ([src/data/pecan_street.py](src/data/pecan_street.py)) | Done |
| Unit tests (`tests/test_did.py`, `test_simulate.py`, `test_london_loader.py`) | Done |
| Pre-commit hooks (file size + pytest) | Done |
| Data docs ([data/README.md](data/README.md)) — LCL swap noted | Done |

---

## Data situation

| Source | Role | Status |
|---|---|---|
| **Pecan Street Dataport** | Intended for Notebook 1 | Stub only — requires registration |
| **Low Carbon London** | Open substitute (dToU vs flat 2013) | Loader + manual zip download (auto URL may 404) |
| **Simulated panel** | Day 1 DiD toy | Used in tests + notebook |

See [data/README.md](data/README.md) for download commands. CI uses [tests/fixtures/lcl_sample.csv](tests/fixtures/lcl_sample.csv) only.

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
├── data/README.md
├── notebooks/00_did_toy_warmup.ipynb
├── src/causal/simulate.py
├── src/causal/did.py
├── src/data/london_smartmeter.py
├── src/data/pecan_street.py
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

# Optional: download one LCL sample block + tariffs (gitignored)
python -c "from src.data.london_smartmeter import download_lcl_sample, download_lcl_tariffs; download_lcl_sample(); download_lcl_tariffs()"
```

---

## Core module API

### Simulator — `src/causal/simulate.py`

| Symbol | Purpose |
|---|---|
| `SimulationConfig` | Household counts, days, ATT (kW), weather jump, peak hours |
| `simulate_residential_load()` | Hourly panel with treated/control, post flag, confounder |
| `summarize_panel()` | Row/household counts for sanity checks |

### DiD — `src/causal/did.py`

| Symbol | Purpose |
|---|---|
| `estimate_did(frame, peak_hours=...)` | Two-way OLS; returns `DidResult` with ATT, SE, 95% CI |
| `naive_before_after(frame, peak_hours=...)` | Treated-only before/after (confounded in toy) |
| `DidResult` | `att_kw`, `std_error`, `ci_low`, `ci_high`, `n_obs` |

### Data — `src/data/`

| Symbol | Purpose |
|---|---|
| `load_lcl_sample()` | Load committed LCL fixture |
| `load_lcl_csv(path)` | Parse LCL CSV → normalized columns + `load_kw` |
| `download_lcl_sample()` / `download_lcl_tariffs()` | Fetch sample block + tariff schedule |
| `load_pecan_street()` | Stub — raises until Dataport files exist |

---

## Suggested next step (Day 2+)

1. **Synthetic control on LCL** — treated dToU vs control flat-rate; build counterfactual peak-load path
2. **Propensity / DoWhy** — document identification graph for opt-in tariff assignment
3. **MAPIE + LightGBM** — port Week 3 forecaster; rolling backtest with empirical coverage

---

## Auth / environment

- Remote: `https://github.com/mehmetertac/energy-causal-conformal.git`
- Python venv at `.venv/` (gitignored)
- GPyTorch in requirements for later GP solar example; Day 1 tests do not import it
