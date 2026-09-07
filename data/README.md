# Data sources

## Intended: Pecan Street Dataport

The capstone **Notebook 1** (synthetic control on a tariff-style intervention) was originally scoped for [Pecan Street Dataport](https://www.pecanstreet.org/dataport/) residential load (`dataid`, `localminute`, `use`).

Access requires registration and approval. Place downloaded CSVs in:

```
data/raw/pecan/
```

Use [`src/data/pecan_street.py`](../src/data/pecan_street.py) once files are available. Until then the stub raises `PecanStreetNotAvailableError` with setup instructions.

---

## Day 1 substitute: Low Carbon London (LCL)

We use the open [SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/) dataset:

- ~5,567 households, half-hourly kWh, Nov 2011 – Feb 2014
- **~1,122 households** on a **dynamic time-of-use (dToU) tariff in 2013** (treated)
- **~4,500 households** on a flat rate (control)

This is the natural DiD / synthetic-control setting for "Did the tariff reduce peak load?"

### Download (sample — recommended for development)

London Datastore often **does not expose direct CSV URLs** (automated download may 404). Manual steps:

1. Open [SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/)
2. Download **low-carbon-london-data-168-files** (758 MB zip) and/or **Tariffs**
3. Extract `block_0.csv` and `tariffs.csv` into `data/raw/lcl/`

Optional helper (works when Datastore URLs are live):

```powershell
python -c "from src.data.london_smartmeter import download_lcl_sample, download_lcl_tariffs; download_lcl_sample(); download_lcl_tariffs()"
```

Files land in `data/raw/lcl/` (gitignored):

| File | Purpose |
|---|---|
| `block_0.csv` | One split block (~1M rows) from the 168-file pack |
| `tariffs.csv` | dToU price schedule for 2013 |

### Full dataset (optional)

The London Datastore also hosts a single ~764 MB zip (~10 GB unzipped, ~167M rows). Only pull this when you need full-city coverage.

### Tests

CI uses the committed fixture [`tests/fixtures/lcl_sample.csv`](../tests/fixtures/lcl_sample.csv) — no network required.

---

## Git policy

- `data/raw/` and `data/processed/` are **gitignored**
- Never commit household-level smart-meter extracts
