# database.py
import sqlite3
import datetime

DB_NAME = "therapy_sessions.db"

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Main table for a therapy series
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS therapy_series (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_key TEXT NOT NULL,
        llm_provider TEXT NOT NULL,
        total_sessions INTEGER,
        session_duration_minutes INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table to store individual messages for each session
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        series_id INTEGER NOT NULL,
        session_number INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (series_id) REFERENCES therapy_series (id)
    )
    """)
    
    # Table to store evaluations for each session
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        series_id INTEGER NOT NULL,
        session_number INTEGER NOT NULL,
        report TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (series_id) REFERENCES therapy_series (id)
    )
    """)
    
    conn.commit()
    conn.close()

def start_new_therapy_series(patient_key, llm_provider):
    """Creates a new record for a therapy series and returns its ID."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO therapy_series (patient_key, llm_provider) VALUES (?, ?)",
        (patient_key, llm_provider)
    )
    series_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return series_id

def save_session_plan(series_id, total_sessions, duration_minutes):
    """Updates a therapy series with the session plan."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE therapy_series SET total_sessions = ?, session_duration_minutes = ? WHERE id = ?",
        (total_sessions, duration_minutes, series_id)
    )
    conn.commit()
    conn.close()

def save_message(series_id, session_number, role, content):
    """Saves a single chat message to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (series_id, session_number, role, content) VALUES (?, ?, ?, ?)",
        (series_id, session_number, role, content)
    )
    conn.commit()
    conn.close()

def save_evaluation(series_id, session_number, report):
    """Saves a session evaluation report to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO evaluations (series_id, session_number, report) VALUES (?, ?, ?)",
        (series_id, session_number, report)
    )
    conn.commit()
    conn.close()

def get_all_series():
    """Retrieves all therapy series, both ongoing and completed."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            ts.*, 
            (SELECT COUNT(DISTINCT session_number) FROM evaluations e WHERE e.series_id = ts.id) as completed_sessions
        FROM therapy_series ts
        ORDER BY ts.created_at DESC
    """)
    series_list = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return series_list

def get_series_details(series_id):
    """Gets details for a specific therapy series."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM therapy_series WHERE id = ?", (series_id,))
    details = dict(cursor.fetchone())
    conn.close()
    return details

def get_messages_for_series(series_id):
    """Retrieves all messages for a given therapy series, ordered correctly."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE series_id = ? ORDER BY timestamp ASC",
        (series_id,)
    )
    messages = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return messages

def get_latest_session_number(series_id):
    """Finds the number of the last session that has messages."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    result = cursor.execute(
        "SELECT MAX(session_number) FROM messages WHERE series_id = ?",
        (series_id,)
    ).fetchone()[0]
    conn.close()
    return result if result is not None else 0