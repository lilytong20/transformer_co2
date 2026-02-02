# PostgreSQL schema and query helpers.

import os
import json
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://app:changeme@localhost:5432/co2_sensor"
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    experiment_id VARCHAR(20),
    sampling_point INT CHECK (sampling_point BETWEEN 1 AND 6),
    features JSONB NOT NULL,
    co2_measured FLOAT
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50),
    input_window INT,
    forecast_window INT,
    predictions JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
"""


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def insert_reading(timestamp, experiment_id, sampling_point, features, co2_measured=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sensor_readings
                   (timestamp, experiment_id, sampling_point, features, co2_measured)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (timestamp, experiment_id, sampling_point, json.dumps(features), co2_measured),
            )
            return cur.fetchone()[0]


def insert_prediction(timestamp, model_version, input_window, forecast_window, predictions):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                   (timestamp, model_version, input_window, forecast_window, predictions)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (timestamp, model_version, input_window, forecast_window, json.dumps(predictions)),
            )
            return cur.fetchone()[0]


def get_predictions(start=None, end=None, limit=100):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []
            if start:
                query += " AND timestamp >= %s"
                params.append(start)
            if end:
                query += " AND timestamp <= %s"
                params.append(end)
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            return cur.fetchall()
