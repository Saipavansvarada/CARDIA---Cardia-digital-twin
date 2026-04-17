"""
CARDIA — Database
SQLite storage for all session readings.
"""

import sqlite3
import os
from datetime import datetime
from config import PATHS

os.makedirs("data", exist_ok=True)
DB_FILE = PATHS["database"]


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hr        REAL,
            hrv       REAL,
            load      REAL,
            sleep     REAL,
            spo2      REAL,
            ces       REAL,
            strategy  TEXT,
            anomaly   INTEGER,
            zone      INTEGER,
            safety    TEXT,
            coach     TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_reading(hr, hrv, load, sleep, spo2, ces,
                   strategy, anomaly, zone, safety, coach=""):
    """Insert one reading into the database."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO readings
        (timestamp,hr,hrv,load,sleep,spo2,ces,strategy,anomaly,zone,safety,coach)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        hr, hrv, load, sleep, spo2, ces,
        strategy, anomaly, zone, safety, coach
    ))
    conn.commit()
    conn.close()


def get_recent(limit=50):
    """Get most recent readings."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT * FROM readings
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    cols = ["id","timestamp","hr","hrv","load","sleep",
            "spo2","ces","strategy","anomaly","zone","safety","coach"]
    return [dict(zip(cols, r)) for r in rows]


def get_stats():
    """Get session statistics."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*)        as total,
            AVG(hr)         as avg_hr,
            AVG(hrv)        as avg_hrv,
            AVG(ces)        as avg_ces,
            MIN(ces)        as min_ces,
            MAX(ces)        as max_ces
        FROM readings
    """)
    row  = c.fetchone()
    conn.close()
    if row and row[0]:
        return {
            "total":   row[0],
            "avg_hr":  round(row[1] or 0, 1),
            "avg_hrv": round(row[2] or 0, 1),
            "avg_ces": round(row[3] or 0, 1),
            "min_ces": round(row[4] or 0, 1),
            "max_ces": round(row[5] or 0, 1),
        }
    return {}


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_FILE}")