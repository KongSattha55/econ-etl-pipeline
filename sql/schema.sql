-- econ_pipeline schema
-- Run: psql -d econ_pipeline -f sql/schema.sql

CREATE TABLE IF NOT EXISTS countries (
    id           SERIAL PRIMARY KEY,
    iso_code     CHAR(3)      NOT NULL UNIQUE,
    name         VARCHAR(100) NOT NULL,
    region       VARCHAR(100),
    income_group VARCHAR(50),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indicators (
    id           SERIAL PRIMARY KEY,
    country_id   INT          NOT NULL REFERENCES countries(id) ON DELETE CASCADE,
    indicator    VARCHAR(100) NOT NULL,
    source       VARCHAR(100),
    year         SMALLINT     NOT NULL,
    value        NUMERIC(18,4),
    unit         VARCHAR(50),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_id, indicator, source, year)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    country_id      INT          NOT NULL REFERENCES countries(id) ON DELETE CASCADE,
    indicator       VARCHAR(100) NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    predicted_year  SMALLINT     NOT NULL,
    predicted_value NUMERIC(18,4),
    confidence_low  NUMERIC(18,4),
    confidence_high NUMERIC(18,4),
    run_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (country_id, indicator, model_name, predicted_year)
);

CREATE INDEX IF NOT EXISTS idx_indicators_country   ON indicators(country_id);
CREATE INDEX IF NOT EXISTS idx_indicators_indicator ON indicators(indicator);
CREATE INDEX IF NOT EXISTS idx_indicators_year      ON indicators(year);
CREATE INDEX IF NOT EXISTS idx_predictions_country  ON predictions(country_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model    ON predictions(model_name);
