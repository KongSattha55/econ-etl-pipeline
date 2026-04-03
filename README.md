# econ-etl-pipeline

ETL pipeline for economic and governance indicators — extracts data from public sources (World Bank, IMF, Transparency International), loads it into PostgreSQL, and runs predictive models.

## Schema

```
countries    — master list (iso_code, name, region, income_group)
indicators   — raw time-series data (country_id → indicator, source, year, value)
predictions  — model output (country_id → indicator, model_name, predicted_year, value + CI)
```

## Setup

```bash
cp .env.example .env          # fill in DB credentials
psql -d econ_pipeline -f sql/schema.sql
pip install -r requirements.txt
```

## Structure

```
etl/          extract.py · transform.py · load.py
sql/          schema.sql
notebooks/    exploratory analysis
tests/        unit + integration tests
Dataset/      raw source files
```
