import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import time

DEFAULT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS architectures (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  blueprint_json TEXT,
  created_at REAL DEFAULT (strftime('%s','now')),
  params INTEGER,
  flops INTEGER,
  summary TEXT
);

CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  arch_id INTEGER,
  run_id TEXT,
  config_json TEXT,
  start_ts REAL,
  end_ts REAL,
  checkpoint_path TEXT,
  FOREIGN KEY(arch_id) REFERENCES architectures(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trial_id INTEGER,
  epoch INTEGER,
  val_acc REAL,
  val_loss REAL,
  latency_cpu_ms REAL,
  latency_cuda_ms REAL,
  created_at REAL DEFAULT (strftime('%s','now')),
  FOREIGN KEY(trial_id) REFERENCES trials(id)
);

CREATE INDEX IF NOT EXISTS idx_arch_params ON architectures(params);
"""

class DKBClient:
    def __init__(self, path: str = "dkb.sqlite"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self._conn.cursor()
        cur.executescript(DEFAULT_SCHEMA)
        self._conn.commit()

    def add_architecture(self, name: str, blueprint_json: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> int:
        cur = self._conn.cursor()
        bj = json.dumps(blueprint_json)
        params = features.get("params") if features else None
        flops = features.get("flops") if features else None
        summary = features.get("summary") if features else None
        cur.execute(
            "INSERT INTO architectures (name, blueprint_json, params, flops, summary) VALUES (?, ?, ?, ?, ?)",
            (name, bj, params, flops, summary)
        )
        self._conn.commit()
        return cur.lastrowid

    def get_architecture(self, arch_id: int) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM architectures WHERE id = ?", (arch_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def add_trial(self, arch_id: int, run_id: str, config_json: Dict[str, Any], checkpoint_path: Optional[str] = None, start_ts: Optional[float]=None, end_ts: Optional[float]=None) -> int:
        cur = self._conn.cursor()
        cj = json.dumps(config_json)
        start_ts = start_ts or time.time()
        cur.execute(
            "INSERT INTO trials (arch_id, run_id, config_json, start_ts, end_ts, checkpoint_path) VALUES (?, ?, ?, ?, ?, ?)",
            (arch_id, run_id, cj, start_ts, end_ts, checkpoint_path)
        )
        self._conn.commit()
        return cur.lastrowid

    def add_metrics(self, trial_id: int, epoch: int, val_acc: float, val_loss: float, latency_cpu_ms: Optional[float], latency_cuda_ms: Optional[float]):
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO metrics (trial_id, epoch, val_acc, val_loss, latency_cpu_ms, latency_cuda_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (trial_id, epoch, val_acc, val_loss, latency_cpu_ms, latency_cuda_ms)
        )
        self._conn.commit()
        return cur.lastrowid

    def query_architectures(self, min_params: Optional[int]=None, max_params: Optional[int]=None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM architectures WHERE 1=1"
        args: List[Any] = []
        if min_params is not None:
            q += " AND params >= ?"; args.append(min_params)
        if max_params is not None:
            q += " AND params <= ?"; args.append(max_params)
        cur = self._conn.cursor()
        cur.execute(q, tuple(args))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def latest_trials_for_arch(self, arch_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM trials WHERE arch_id = ? ORDER BY start_ts DESC LIMIT ?", (arch_id, limit))
        return [dict(r) for r in cur.fetchall()]

    def get_metrics_for_trial(self, trial_id: int) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM metrics WHERE trial_id = ? ORDER BY epoch ASC", (trial_id,))
        return [dict(r) for r in cur.fetchall()]

    def close(self):
        try:
            self._conn.commit()
        finally:
            self._conn.close()
