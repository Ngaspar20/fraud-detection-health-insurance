import sqlite3
import pandas as pd
import hashlib
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "claims.db"


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            filename     TEXT,
            uploaded_at  TEXT,
            row_count    INTEGER,
            columns      TEXT
        )
    """)
    conn.commit()
    return conn


def save_session(df: pd.DataFrame, filename: str) -> str:
    session_id = hashlib.md5(f"{filename}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    conn = _get_conn()
    df.to_sql(f"claims_{session_id}", conn, if_exists="replace", index=False)
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?)",
        (session_id, filename, datetime.now().isoformat(), len(df), ",".join(df.columns.tolist()))
    )
    conn.commit()
    conn.close()
    return session_id


def list_sessions() -> pd.DataFrame:
    try:
        conn = _get_conn()
        df = pd.read_sql("SELECT * FROM sessions ORDER BY uploaded_at DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_session(session_id: str) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql(f"SELECT * FROM claims_{session_id}", conn)
    conn.close()
    return df


def delete_session(session_id: str):
    conn = _get_conn()
    conn.execute(f"DROP TABLE IF EXISTS claims_{session_id}")
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def parse_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, low_memory=False)
    else:
        df = pd.read_excel(uploaded_file)
    return df
