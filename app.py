"""
AI-Powered Machine Diagnostics Analyzer
========================================
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
import libsql_client.dbapi as dbapi

# Optional PDF generation - handle missing reportlab gracefully
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be limited. Install with: `pip install reportlab`")

# Standard fault labels for the dropdown menu
FAULT_LABELS = [
    "Valve Leakage", "Valve Wear", "Valve Sticking or Fouling",
    "Valve Impact or Slamming", "Broken or Missing Valve Parts",
    "Valve Misalignment", "Spring Fatigue or Failure", "Other"
]


# --- Database Setup ---

def init_db():
    """Initializes the database connection using Turso."""
    try:
        url = st.secrets["TURSO_DATABASE_URL"]
        auth_token = st.secrets["TURSO_AUTH_TOKEN"]
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()

    try:
        conn = dbapi.connect(database=url, auth_token=auth_token)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, machine_id TEXT, rpm TEXT)
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, anomaly_count INTEGER, threshold REAL, FOREIGN KEY (session_id) REFERENCES sessions (id))
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, analysis_id INTEGER, label_text TEXT, FOREIGN KEY (analysis_id) REFERENCES analyses (id))
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, analysis_id INTEGER, event_type TEXT, crank_angle REAL, FOREIGN KEY (analysis_id) REFERENCES analyses (id))
        ''')
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()


# --- Session State Initialization ---
db_conn = init_db()
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = {}

# --- Helper Functions ---
def safe_db_operation(operation, *args):
    """Safely execute database operations."""
    try:
        cursor = db_conn.cursor()
        result = cursor.execute(operation, args)
        db_conn.commit()
        return result
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def get_last_row_id(cursor):
    """Get the last inserted row ID, compatible with Turso."""
    return cursor.lastrowid

@st.cache_data
def load_all_curves_data(_curves_xml_content):
    """Parses the entire Curves.xml file once and caches the resulting DataFrame."""
    try:
        root = ET.fromstring(_curves_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next(ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves')
        table = ws.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]
        data = [[cell.find('ss:Data', NS).text for cell in r.findall('ss:Cell', NS)] for r in rows[6:]]
        if not data: return None, None
        df = pd.DataFrame(data, columns=full_header_list[:len(data[0])]).apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)
        return df, full_header_list[:len(data[0])]
    except Exception as e:
        st.error(f"Failed to load or parse curves data: {e}")
        return None, None

def generate_cylinder_view(df, cylinder_config, envelope_view, vertical_offset, analysis_ids):
    pressure_curve = cylinder_config.get('pressure_curve')
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []

    pres_mean, pres_std = df[pressure_curve].mean(), df[pressure_curve].std()
    pres_thresh = pres_mean + 2 * pres_std
    df[f'{pressure_curve}_anom'] = df[pressure_curve] > pres_thresh
    report_data.append({"name": "Pressure", "curve_name": pressure_curve, "threshold": pres_thresh, "count": df[f'{pressure_curve}_anom'].sum(), "unit": "PSI"})

    for vc in valve_curves:
        curve_name = vc['curve']
        vib_mean, vib_std = df[curve_name].mean(), df[curve_name].std()
        vib_thresh = vib_mean + 2 * vib_std
        df[f'{curve_name}_anom'] = df[curve_name] > vib_thresh
        report_data.append({"name": vc['name'], "curve_name": curve_name, "threshold": vib_thresh, "count": df[f'{curve_name}_anom'].sum(), "unit": "G"})

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Crank Angle'], y=df[pressure_curve], name='Pressure (PSI)', line=dict(color='black', width=2)), secondary_y=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(valve_curves)))
    cursor = db_conn.cursor()
    current_offset = 0

    for i, vc in enumerate(valve_curves):
        curve_name = vc['curve']
        label_name = vc['name']
        color_rgba = f'rgba({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255},0.4)'
        
        upper_bound = df[curve_name] + current_offset
        lower_bound = -df[curve_name] + current_offset
        fig.add_trace(go.Scatter(x=df['Crank Angle'], y=upper_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), showlegend=False, hoverinfo='none'), secondary_y=True)
        fig.add_trace(go.Scatter(x=df['Crank Angle'], y=lower_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), fill='tonexty', fillcolor=color_rgba, name=label_name, hoverinfo='none'), secondary_y=True)

        analysis_id = analysis_ids.get(vc['name'])
        if analysis_id:
            events_raw = cursor.execute("SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?", (analysis_id,)).fetchall()
            events = {etype: angle for etype, angle in events_raw}
            if 'open' in events and 'close' in events:
                fig.add_vrect(x0=events['open'], x1=events['close'], fillcolor=color_rgba, layer="below", line_width=0)
            for event_type, crank_angle in events.items():
                fig.add_vline(x=crank_angle, line_width=2, line_dash="dash", line_color='green' if event_type == 'open' else 'red')
        
        current_offset += vertical_offset
    
    fig.update_layout(title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}", xaxis_title="Crank Angle (deg)", template="ggplot2")
    fig.update_yaxes(title_text="<b>Pressure (PSI)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", secondary_y=True)
    return fig, report_data

# --- Main App ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")
st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")

# Sidebar
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader("Upload Curves, Levels, Source XML files", type=["xml"], accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
    if st.button("Start New Analysis / Clear Files"):
        st.session_state.file_uploader_key += 1
        st.session_state.active_session_id = None
        st.rerun()
    st.header("2. View Options")
    envelope_view = st.checkbox("Enable Envelope View", value=True)
    vertical_offset = st.slider("Vertical Offset", 0.0, 5.0, 1.0, 0.1)

if uploaded_files and len(uploaded_files) == 3:
    files_content = {f.name.lower().split('.')[0]: f.getvalue().decode('utf-8') for f in uploaded_files}
    if 'curves' in files_content and 'levels' in files_content and 'source' in files_content:
        df, actual_curve_names = load_all_curves_data(files_content['curves'])
        if df is not None:
            # Continue with your main logic here, using df and other variables
            st.success("Files processed successfully!")
            # Example of what would follow
            # discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
            # if discovered_config:
            #     # ... and so on
            pass # Placeholder for the rest of your app logic
else:
    st.info("Please upload all three required XML files to begin.")

# Displaying the plot (example)
# if 'df' in locals():
#     # This is just a placeholder to show how you would call the plotting
#     # You need to replace this with your actual logic to get the selected cylinder config etc.
#     # fig, report_data = generate_cylinder_view(df, example_config, envelope_view, vertical_offset, {})
#     # st.plotly_chart(fig, use_container_width=True)
#     pass
