import sqlite3
import json
import time
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from contextlib import contextmanager

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;  -- Better concurrency

CREATE TABLE IF NOT EXISTS architectures (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE,         -- Prevent duplicate names
  blueprint_json TEXT,
  created_at REAL DEFAULT (strftime('%s','now')),
  params INTEGER,
  flops INTEGER,
  summary TEXT
);

CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  arch_id INTEGER,
  run_id TEXT UNIQUE,
  config_json TEXT,
  start_ts REAL,
  end_ts REAL,
  status TEXT DEFAULT 'PENDING',  -- PENDING, RUNNING, COMPLETED, FAILED
  best_val_acc REAL DEFAULT 0.0,  -- Cached for fast sorting
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
CREATE INDEX IF NOT EXISTS idx_trials_best_acc ON trials(best_val_acc DESC);
"""

class DKBClient:
    def __init__(self, path: str = "dkb.sqlite"):
        self.path = str(path)
        self._conn = None
        
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()

    def _ensure_schema(self):
        try:
            cur = self._conn.cursor()
            cur.executescript(SCHEMA)
            self._conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Schema initialization failed: {e}")

    def close(self):
        if self._conn:
            try:
                self._conn.commit()
            except Exception:
                pass
            self._conn.close()
            self._conn = None

    def add_architecture(self, name: str, blueprint_json: Dict, features: Optional[Dict] = None) -> int:
        self.connect()
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM architectures WHERE name = ?", (name,))
        existing = cur.fetchone()
        if existing:
            return existing['id']

        bj = json.dumps(blueprint_json)
        feats = features or {}
        cur.execute(
            "INSERT INTO architectures (name, blueprint_json, params, flops, summary) VALUES (?, ?, ?, ?, ?)",
            (name, bj, feats.get("params"), feats.get("flops"), feats.get("summary"))
        )
        self._conn.commit()
        return cur.lastrowid

    def add_trial(self, arch_id: int, run_id: str, config: Dict, start_ts: float = None) -> int:
        self.connect()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO trials (arch_id, run_id, config_json, start_ts, status) VALUES (?, ?, ?, ?, 'RUNNING')",
            (arch_id, run_id, json.dumps(config), start_ts or time.time())
        )
        self._conn.commit()
        return cur.lastrowid

    def update_trial_result(self, trial_id: int, status: str, best_acc: float = None, ckpt_path: str = None):
        """Updates the trial summary when training finishes."""
        self.connect()
        updates = ["status = ?", "end_ts = ?"]
        params = [status, time.time()]
        
        if best_acc is not None:
            updates.append("best_val_acc = ?")
            params.append(best_acc)
        if ckpt_path:
            updates.append("checkpoint_path = ?")
            params.append(ckpt_path)
            
        params.append(trial_id) 
        
        query = f"UPDATE trials SET {', '.join(updates)} WHERE id = ?"
        self._conn.execute(query, tuple(params))
        self._conn.commit()

    def add_metrics(self, trial_id: int, epoch: int, metrics: Dict[str, float]):
        self.connect()
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO metrics 
               (trial_id, epoch, val_acc, val_loss, latency_cpu_ms, latency_cuda_ms) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (trial_id, epoch, metrics.get("val_acc"), metrics.get("val_loss"), 
             metrics.get("latency_cpu_ms"), metrics.get("latency_cuda_ms"))
        )
        if metrics.get("val_acc"):
            cur.execute("""
                UPDATE trials SET best_val_acc = MAX(best_val_acc, ?) WHERE id = ?
            """, (metrics["val_acc"], trial_id))
            
        self._conn.commit()

    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        self.connect()
        query = """
        SELECT a.name, a.params, a.flops, t.best_val_acc, t.status, t.id as trial_id
        FROM trials t
        JOIN architectures a ON t.arch_id = a.id
        WHERE t.status = 'COMPLETED'
        ORDER BY t.best_val_acc DESC
        LIMIT ?
        """
        cur = self._conn.cursor()
        cur.execute(query, (limit,))
        return [dict(r) for r in cur.fetchall()]

    def get_pareto_candidates(self) -> List[Dict]:
        """
        Fetches data for plotting the Pareto Frontier (Accuracy vs Efficiency).
        Returns [ {acc, params, flops, latency} ... ]
        """
        self.connect()
        query = """
        SELECT a.id, a.params, a.flops, t.best_val_acc, 
               (SELECT AVG(latency_cpu_ms) FROM metrics m WHERE m.trial_id = t.id) as avg_latency
        FROM trials t
        JOIN architectures a ON t.arch_id = a.id
        WHERE t.best_val_acc > 0
        """
        cur = self._conn.cursor()
        cur.execute(query)
        return [dict(r) for r in cur.fetchall()]