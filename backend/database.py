import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "epidemic_dashboard.db")

def init_db():
    """Initialize the SQLite database and create necessary tables if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for storing baseline predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            year INTEGER,
            week INTEGER,
            censuscode INTEGER,
            district_name TEXT,
            state TEXT,
            prob REAL,
            pred_cases REAL,
            actual_cases INTEGER,
            truth INTEGER,
            is_simulated INTEGER DEFAULT 0
        )
    """)
    
    # Table for storing What-If counterfactual simulation logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS what_if_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            censuscode INTEGER,
            district_name TEXT,
            temp_shift REAL,
            preci_shift REAL,
            cases_shift REAL,
            symptoms_shift REAL,
            simulated_prob REAL,
            simulated_cases REAL
        )
    """)
    
    conn.commit()
    conn.close()

def log_prediction(year, week, censuscode, district_name, state, prob, pred_cases, actual_cases, truth, is_simulated=0):
    """Insert a prediction record into the database, ignoring duplicates for the same time window and district."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 FROM predictions 
        WHERE year = ? AND week = ? AND censuscode = ? AND is_simulated = ?
    """, (year, week, censuscode, is_simulated))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute("""
            INSERT INTO predictions (year, week, censuscode, district_name, state, prob, pred_cases, actual_cases, truth, is_simulated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (year, week, censuscode, district_name, state, prob, pred_cases, actual_cases, truth, is_simulated))
        conn.commit()
    conn.close()

def log_what_if(censuscode, district_name, temp_shift, preci_shift, cases_shift, symptoms_shift, simulated_prob, simulated_cases):
    """Insert a What-If simulation log into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO what_if_history (censuscode, district_name, temp_shift, preci_shift, cases_shift, symptoms_shift, simulated_prob, simulated_cases)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (censuscode, district_name, temp_shift, preci_shift, cases_shift, symptoms_shift, simulated_prob, simulated_cases))
    conn.commit()
    conn.close()

def get_predictions_history(limit=100):
    """Retrieve the latest prediction records."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_what_if_history(limit=100):
    """Retrieve the latest What-If simulation logs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM what_if_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
