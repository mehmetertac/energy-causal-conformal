# AGENT.md — rules for AI agents

Guidance for agents working in **energy-causal-conformal**. Read this file first, then follow the linked docs for domain context.

---

## Documentation map

| Doc | Purpose |
|---|---|
| [README.md](README.md) | Project goal, setup, two-notebook framing, tariff narrative |
| [handover.md](handover.md) | Current status, repo layout, module API, week roadmap |
| [docs/causal_mental_model.md](docs/causal_mental_model.md) | Potential outcomes, DiD assumptions, energy examples |
| [data/README.md](data/README.md) | Pecan Street intent, LCL substitute, download commands |
| [pytest.ini](pytest.ini) | Test discovery (`tests/`, `pythonpath = .`) |
| [requirements.txt](requirements.txt) | Dependencies (DoWhy, EconML, MAPIE, GPyTorch, LightGBM) |
| [notebooks/](notebooks/) | Exploratory walkthroughs |
| [src/causal/](src/causal/) | DiD toy simulator and estimator |
| [src/data/](src/data/) | LCL loader + Pecan Street stub |
| [tests/](tests/) | Unit tests (synthetic data; no network in CI) |

Headline themes for this repo: **causal effects with honest uncertainty**, and **conformal coverage guarantees** — not just point forecasts.

---

## Rules

### 1. File size limit

- **No file should exceed 1,000 lines.**
- If a file approaches or exceeds that limit, **stop and suggest a refactor** before adding more code (split modules, extract helpers, move tests/notebooks out).
- Pre-commit runs [`scripts/check_file_size.py`](scripts/check_file_size.py) to enforce this.

### 2. Documentation before every push

- **Update documentation before every push** to the repository.
- At minimum, check whether these need updates for your change:
  - [README.md](README.md) — run commands, structure, user-facing behavior
  - [handover.md](handover.md) — done/next steps, API table, artifacts

### 2a. Update handover.md on every push (required)

- **[handover.md](handover.md) must be updated before every push**, even for small changes.
- On each push, at minimum refresh:
  - **Last updated** date at the top
  - **What is done** table (add/mark items)
  - **Repo layout** if files or directories changed
  - **Core module API** if public symbols, CLI flags, or defaults changed
  - **Suggested next step** if priorities shifted
  - **Key commit** hash and one-line summary for the work being pushed
- Do not push without reviewing [handover.md](handover.md) — stale handover is a blocker.

### 3. Tests — always, at least minimal

- **Always create at least minimal unit tests**, even for small changes.
- Add **integration** tests when wiring multiple modules (e.g. loader → causal estimator → notebook export).
- Add **functional** tests when the project supports them (CLI smoke tests, end-to-end with tiny fixtures).
- Existing pattern: [tests/test_did.py](tests/test_did.py) uses simulated load so CI does not depend on smart-meter downloads.
- New causal or conformal logic should get numeric/assertion checks, not only "runs without error."

### 4. Run tests before commit or push

- **Run the test suite before commit or push:**
  ```powershell
  pytest tests/ -q
  ```
- Pre-commit hooks run file-size check + pytest (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

**Git hooks:** Install once after creating the venv:

```powershell
pip install -r requirements.txt
pre-commit install
```

### 5. Keep reading in-repo docs

- Do not guess API or roadmap from memory — use [handover.md](handover.md) for status and [README.md](README.md) for how to run.
- Match existing conventions in [src/causal/did.py](src/causal/did.py) (dataclasses, statsmodels OLS, explicit column names) when extending causal code.
- Conformal evaluation (later) must report **empirical coverage** on rolling backtests, not nominal coverage alone.

---

## Quick checklist (before push)

- [ ] No file > 1,000 lines (or refactor proposed)
- [ ] [README.md](README.md) updated if behavior or layout changed
- [ ] [handover.md](handover.md) updated (required on every push — date, status, API, key commit)
- [ ] New/changed logic has tests in [tests/](tests/)
- [ ] `pytest tests/ -q` passes
- [ ] `pre-commit run --all-files` passes (optional but recommended)
