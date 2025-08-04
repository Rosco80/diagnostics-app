"""
AI-Powered Machine Diagnostics Analyzer
"""
import streamlit as st
import xml.etree.ElementTree as ET
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import sqlite3
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import libsql_client # Correct import for Turso/libSQL

# --- Configuration ---
FAULT_LABELS = [
    "Valve Leakage", "Valve Wear", "Valve Sticking or Fouling",
    "Valve Impact or Slamming", "Broken or Missing Valve Parts",
    "Valve Misalignment", "Spring Fatigue or Failure", "Other"
]

# --- Database Setup ---
@st.cache_resource
def init_db():
    """Initializes the database connection using Turso."""
    try:
        url = st.secrets["TURSO_DATABASE_URL"]
        auth_token = st.secrets["TURSO_AUTH_TOKEN"]
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()

    try:
        # Use the create_client method
        client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
        
        # Create tables if they don't exist
        client.batch([
            "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, machine_id TEXT, rpm TEXT)",
            "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, anomaly_count INTEGER, threshold REAL, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, analysis_id INTEGER, label_text TEXT, FOREIGN KEY (analysis_id) REFERENCES analyses (id))",
            "CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, analysis_id INTEGER, event_type TEXT, crank_angle REAL, FOREIGN KEY (analysis_id) REFERENCES analyses (id))"
        ])
        return client
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()

# --- Helper Functions ---
def get_last_row_id(client):
    """Get the last inserted row ID for Turso."""
    # Turso client does not directly support lastrowid in the same way.
    # We will query for the max ID as a workaround. This is not perfectly atomic
    # but is sufficient for this single-user application.
    rs = client.execute("SELECT last_insert_rowid()")
    return rs.rows[0][0]


# --- Main App & UI ---

st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")
st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")

# Initialize database connection
db_client = init_db()

# Initialize session state
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# --- The rest of your application code goes here ---
# Make sure to replace all instances of `db_conn.cursor()` or `db_conn.execute`
# with `db_client.execute`. The client object acts as both.
# Also replace `cursor.lastrowid` with `get_last_row_id(db_client)`

# Example of how to adapt your existing code:
# OLD: cursor = db_conn.cursor()
# NEW: # No cursor needed, just use db_client directly

# OLD: cursor.execute(...)
# NEW: db_client.execute(...)

# OLD: st.session_state.active_session_id = cursor.lastrowid
# NEW: st.session_state.active_session_id = get_last_row_id(db_client)

# ... (Insert the rest of your app's main logic, sidebar, plotting functions, etc. here,
# making the changes described above)
