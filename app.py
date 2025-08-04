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
import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import libsql_client

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")

# --- Global Configuration & Constants ---
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
        client = libsql_client.create_client_sync(url=url, auth_token=auth_token)
        # Create tables if they don't exist
        client.batch([
            "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, machine_id TEXT, rpm TEXT)",
            "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, session_id INTEGER, cylinder_name TEXT, curve_name TEXT, anomaly_count INTEGER, threshold REAL, FOREIGN KEY (session_id) REFERENCES sessions (id))",
            "CREATE TABLE IF NOT EXISTS labels (id INTEGER PRIMARY KEY, analysis_id INTEGER, label_text TEXT, FOREIGN KEY (analysis_id) REFERENCES analyses (id))",
            "CREATE TABLE IF NOT EXISTS valve_events (id INTEGER PRIMARY KEY, analysis_id INTEGER, event_type TEXT, crank_angle REAL, FOREIGN KEY (analysis_id) REFERENCES analyses (id))"
        ])
        return client
    except KeyError:
        st.error("Database secrets (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to Turso database: {e}")
        st.stop()

# --- Helper Functions ---
def get_last_row_id(_client):
    rs = _client.execute("SELECT last_insert_rowid()")
    return rs.rows[0][0] if rs.rows else None

@st.cache_data
def load_all_curves_data(_curves_xml_content):
    try:
        root = ET.fromstring(_curves_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'), None)
        if ws is None: return None, None
        table = ws.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]
        data = [[cell.find('ss:Data', NS).text for cell in r.findall('ss:Cell', NS)] for r in rows[6:]]
        if not data: return None, None
        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]
        df = pd.DataFrame(data, columns=actual_columns).apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)
        return df, actual_columns
    except Exception as e:
        st.error(f"Failed to load or parse curves data: {e}")
        return None, None

@st.cache_data
def auto_discover_configuration(_source_xml_content, all_curve_names):
    try:
        source_root = ET.fromstring(_source_xml_content)
        num_cyl_str = find_xml_value(source_root, 'Source', "COMPRESSOR NUMBER OF CYLINDERS", 2)
        machine_id = find_xml_value(source_root, 'Source', "Machine", 1)
        if num_cyl_str == "N/A" or int(num_cyl_str) == 0: return None
        num_cylinders = int(num_cyl_str)
        cylinders_config = []
        for i in range(1, num_cylinders + 1):
            pressure_curve = next((c for c in all_curve_names if f".{i}H." in c and "STATIC" in c), None)
            if not pressure_curve:
                pressure_curve = next((c for c in all_curve_names if f".{i}C." in c and "STATIC" in c), None)
            valve_curves = [
                {"name": f"Cyl {i} HE Discharge", "curve": next((c for c in all_curve_names if f".{i}HD" in c and "VIBRATION" in c), None)},
                {"name": f"Cyl {i} HE Suction", "curve": next((c for c in all_curve_names if f".{i}HS" in c and "VIBRATION" in c), None)},
                {"name": f"Cyl {i} CE Discharge", "curve": next((c for c in all_curve_names if f".{i}CD" in c and "VIBRATION" in c), None)},
                {"name": f"Cyl {i} CE Suction", "curve": next((c for c in all_curve_names if f".{i}CS" in c and "VIBRATION" in c), None)}
            ]
            if pressure_curve and any(vc['curve'] for vc in valve_curves):
                cylinders_config.append({"cylinder_name": f"Cylinder {i}", "pressure_curve": pressure_curve, "valve_vibration_curves": [vc for vc in valve_curves if vc['curve']]})
        return {"machine_id": machine_id, "cylinders": cylinders_config}
    except Exception as e:
        st.error(f"An error occurred during auto-discovery: {e}")
        return None

def find_xml_value(root, sheet_name, partial_key, col_offset):
    try:
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
        if ws is None: return "N/A"
        rows = ws.findall('.//ss:Row', NS)
        for row in rows:
            cells = row.findall('ss:Cell', NS)
            if cells and cells[0].find('ss:Data', NS) is not None and partial_key.upper() in (cells[0].find('ss:Data', NS).text or "").upper():
                if len(cells) > col_offset and cells[col_offset].find('ss:Data', NS) is not None:
                    return cells[col_offset].find('ss:Data', NS).text
        return "N/A"
    except Exception:
        return "N/A"


def generate_cylinder_view(_db_client, df, cylinder_config, envelope_view, vertical_offset, analysis_ids):
    pressure_curve = cylinder_config.get('pressure_curve')
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []

    if pressure_curve in df.columns:
        pres_mean, pres_std = df[pressure_curve].mean(), df[pressure_curve].std()
        pres_thresh = pres_mean + 2 * pres_std
        report_data.append({"name": "Pressure", "curve_name": pressure_curve, "threshold": pres_thresh, "count": (df[pressure_curve] > pres_thresh).sum(), "unit": "PSI"})
    for vc in valve_curves:
        curve_name = vc['curve']
        if curve_name in df.columns:
            vib_mean, vib_std = df[curve_name].mean(), df[curve_name].std()
            vib_thresh = vib_mean + 2 * vib_std
            report_data.append({"name": vc['name'], "curve_name": curve_name, "threshold": vib_thresh, "count": (df[curve_name] > vib_thresh).sum(), "unit": "G"})

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Crank Angle'], y=df[pressure_curve], name='Pressure (PSI)', line=dict(color='black', width=2)), secondary_y=False)

    colors = plt.cm.viridis(np.linspace(0, 1, len(valve_curves)))
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
            events_raw = _db_client.execute("SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?", (analysis_id,)).rows
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

# --- Main Application UI and Logic ---
db_client = init_db()

if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")
st.markdown("Upload your machine's XML data files. The configuration will be discovered automatically.")

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
    files_content = {}
    for f in uploaded_files:
        if 'curves' in f.name.lower(): files_content['curves'] = f.getvalue().decode('utf-8')
        elif 'levels' in f.name.lower(): files_content['levels'] = f.getvalue().decode('utf-8')
        elif 'source' in f.name.lower(): files_content['source'] = f.getvalue().decode('utf-8')

    if 'curves' in files_content and 'source' in files_content:
        df, actual_curve_names = load_all_curves_data(files_content['curves'])
        if df is not None:
            discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
            if discovered_config:
                if st.session_state.active_session_id is None:
                    db_client.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (discovered_config.get('machine_id', 'N/A'), "N/A"))
                    st.session_state.active_session_id = get_last_row_id(db_client)
                
                cylinders = discovered_config.get("cylinders", [])
                cylinder_names = [c.get("cylinder_name") for c in cylinders]
                with st.sidebar:
                    selected_cylinder_name = st.selectbox("Select Cylinder for Detailed View", cylinder_names)
                
                selected_cylinder_config = next((c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), None)

                if selected_cylinder_config:
                    _, temp_report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, {})
                    analysis_ids = {}
                    for item in temp_report_data:
                        rs = db_client.execute("SELECT id FROM analyses WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?", (st.session_state.active_session_id, selected_cylinder_name, item['curve_name']))
                        existing_id = rs.rows[0][0] if rs.rows else None
                        if existing_id:
                            analysis_id = existing_id
                            db_client.execute("UPDATE analyses SET anomaly_count = ?, threshold = ? WHERE id = ?", (item['count'], item['threshold'], analysis_id))
                        else:
                            db_client.execute("INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)", (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold']))
                            analysis_id = get_last_row_id(db_client)
                        analysis_ids[item['name']] = analysis_id
                    
                    fig, report_data = generate_cylinder_view(db_client, df.copy(), selected_cylinder_config, envelope_view, vertical_offset, analysis_ids)
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Add labels and mark valve events"):
                        # Your forms for labeling and adding valve events will go here
                        # Remember to use `db_client.execute(...)` for all database operations
                        pass
            else:
                st.error("Could not discover a valid machine configuration.")
        else:
            st.error("Failed to process curve data.")
    else:
        st.warning("Please ensure 'curves' and 'source' XML files are uploaded.")
else:
    st.info("Please upload all three required XML files to begin.")
