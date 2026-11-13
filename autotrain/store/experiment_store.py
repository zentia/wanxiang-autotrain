"""Experiment Store: SQLite-backed experiment and metric records.

M1 scope: experiments table (config snapshot, status) and metrics table
(step-level time series). Query helpers for later stages (hyperparameter
search, agent closed loop).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (experiment_id, step, key)
);
CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics (experiment_id, key);
"""


class ExperimentStore:
    def __init__(self, db_path: str | Path):
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(p))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- experiments --------------------------------------------------

    def create_experiment(self, name: str, config_hash: str, config: dict[str, Any]) -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO experiments (name, config_hash, config_json, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'RUNNING', ?, ?)",
            (name, config_hash, json.dumps(config, sort_keys=True, ensure_ascii=True), now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def set_status(self, experiment_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), experiment_id),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, config_hash, config_json, status, created_at, updated_at "
            "FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "config_hash": row[2],
            "config": json.loads(row[3]),
            "status": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    def find_experiments(self, name: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, name, config_hash, status, created_at FROM experiments WHERE 1=1"
        params: list[Any] = []
        if name is not None:
            sql += " AND name = ?"
            params.append(name)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        rows = self.conn.execute(sql + " ORDER BY id", params).fetchall()
        return [
            {"id": r[0], "name": r[1], "config_hash": r[2], "status": r[3], "created_at": r[4]}
            for r in rows
        ]

    # -- metrics --------------------------------------------------------

    def log_metric(self, experiment_id: int, step: int, key: str, value: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metrics (experiment_id, step, key, value, ts) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, step, key, float(value), time.time()),
        )
        self.conn.commit()

    def log_metrics(self, experiment_id: int, step: int, kv: dict[str, float]) -> None:
        now = time.time()
        self.conn.executemany(
            "INSERT OR REPLACE INTO metrics (experiment_id, step, key, value, ts) VALUES (?, ?, ?, ?, ?)",
            [(experiment_id, step, k, float(v), now) for k, v in kv.items()],
        )
        self.conn.commit()

    def get_metric_series(self, experiment_id: int, key: str) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT step, value FROM metrics WHERE experiment_id = ? AND key = ? ORDER BY step",
            (experiment_id, key),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]
