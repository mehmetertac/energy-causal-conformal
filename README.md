# energy-causal-conformal

Build the **"questions ML alone can't answer" toolkit** — **effects and guarantees**, not just predictions.

Utilities constantly ask causal questions (pilots, tariffs, curtailment counterfactuals). Production risk teams want **distribution-free interval coverage**, not only Bayesian posteriors. This repo makes both explicit and testable.

---

## Two-notebook arc

| Notebook | Question | Method |
|---|---|---|
| **1** (later) | Did the tariff reduce peak residential load? | Synthetic control on smart-meter data |
| **2** (later) | Are forecast intervals trustworthy in production? | MAPIE conformal intervals on a LightGBM load forecast + long rolling backtest |

**Day 1 (now):** causal mental model + DiD toy that recovers a **known injected treatment effect** before touching real meters. See [`notebooks/00_did_toy_warmup.ipynb`](notebooks/00_did_toy_warmup.ipynb).

Concept primer: [`docs/causal_mental_model.md`](docs/causal_mental_model.md)

---

## What this means in tariff / market terms

A causal **average treatment effect (ATT)** on load is not just a kWh number — it translates to:

| Quantity | Formula (illustrative) |
|---|---|
| **Peak MW avoided** | `ATT_kW × n_treated_homes / 1000` |
| **Wholesale energy savings** | `ΔkWh_peak × price_EUR_per_MWh / 1000` |
| **Customer bill delta** | `ΔkWh × tariff_rate` (watch fixed vs volumetric components) |

Day 1 uses **simulated** peak-hour ATT = **−0.4 kW** per treated home. With 50 treated homes that is ~**−0.02 MW** of peak demand removed — small in the toy, but the reporting template scales to real pilots.

**Honest uncertainty:** report confidence intervals on the causal estimate (DiD CI today; conformal **empirical coverage** later — nominal 90% is not enough on its own).

---

## Data

| Source | Role |
|---|---|
| [Pecan Street Dataport](https://www.pecanstreet.org/dataport/) | **Intended** for Notebook 1 (registration required) |
| [Low Carbon London](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/) | **Day 1 substitute** — dToU tariff trial vs flat-rate control, 2013 |

Details: [`data/README.md`](data/README.md)

---

## Quick start

Requires **Python 3.11+** recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pre-commit install

pytest tests/ -q
jupyter notebook notebooks/00_did_toy_warmup.ipynb
```

Optional LCL sample download (gitignored under `data/raw/lcl/`):

```powershell
python -c "from src.data.london_smartmeter import download_lcl_sample, download_lcl_tariffs; download_lcl_sample(); download_lcl_tariffs()"
```

---

## Project structure

```
├── docs/causal_mental_model.md   Causal concepts → energy examples
├── notebooks/                    00_did_toy_warmup.ipynb (Day 1)
├── src/causal/                   DiD simulator + estimator
├── src/data/                     LCL loader, Pecan Street stub
├── tests/                        Unit tests (no network)
├── data/README.md                Data acquisition
├── AGENT.md                      Agent workflow rules
└── handover.md                   Live status for the next session
```

---

## Tools (week roadmap)

| Library | Use |
|---|---|
| DoWhy / EconML | Causal graphs, propensity, heterogeneous effects |
| MAPIE | Conformal prediction intervals |
| LightGBM | Point forecaster baseline (Notebook 2) |
| GPyTorch | Brief GP solar example (compressed; conformal wins in production) |

---

## Agent / contributor workflow

Read [`AGENT.md`](AGENT.md) before making changes. Update [`handover.md`](handover.md) before every push.
