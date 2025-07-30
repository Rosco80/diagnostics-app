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

# --- Database Setup ---

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect('diagnostics.db')
    c = conn.cursor()
    # Session table to track each analysis run
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            machine_id TEXT,
            rpm TEXT
        )
    ''')
    # Analysis table to store results for each cylinder in a session
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            cylinder_name TEXT,
            curve_name TEXT,
            anomaly_count INTEGER,
            threshold REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    # Labels table for supervised learning data
    c.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            label_text TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    conn.commit()
    return conn

# --- Session State Initialization ---
if 'db_conn' not in st.session_state:
    st.session_state.db_conn = init_db()
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None


# --- Helper Functions ---

@st.cache_data
def load_all_curves_data(_curves_xml_content):
    """
    Parses the entire Curves.xml file once and caches the resulting DataFrame.
    """
    try:
        root = ET.fromstring(_curves_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'), None)
        if ws is None:
            st.error("Error: 'Curves' worksheet not found in the XML file.")
            return None, None

        table = ws.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        all_curve_names = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]

        data = []
        for r in rows[6:]:
            cells = r.findall('ss:Cell', NS)
            row_data = [cell.find('ss:Data', NS).text for cell in cells]
            data.append(row_data)
        
        df = pd.DataFrame(data, columns=all_curve_names[:len(data[0])])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)
        return df, all_curve_names
    except Exception as e:
        st.error(f"Failed to load or parse curves data: {e}")
        return None, None


def extract_rpm(_levels_xml_content):
    """Extract machine RPM from the Levels.xml file."""
    try:
        root = ET.fromstring(_levels_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws_levels = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Levels'), None)
        if ws_levels is None: return "N/A"
        
        table = ws_levels.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        for row in rows:
            cells = row.findall('ss:Cell', NS)
            if cells and cells[0].find('ss:Data', NS) is not None:
                cell_text = cells[0].find('ss:Data', NS).text
                if cell_text and "RPM" in cell_text:
                    for cell in cells[1:]:
                         data_node = cell.find('ss:Data', NS)
                         if data_node is not None and data_node.text:
                             return f"{float(data_node.text):.0f}"
    except Exception:
        return "N/A"
    return "N/A"

@st.cache_data
def auto_discover_configuration(_source_xml_content, _curves_xml_content):
    """
    Automatically discovers the machine configuration from the XML files.
    """
    try:
        source_root = ET.fromstring(_source_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        machine_id = ""
        num_cylinders = 0

        rows = source_root.findall('.//ss:Row', NS)
        for row in rows:
            cells = row.findall('ss:Cell', NS)
            if len(cells) > 1 and cells[0].find('ss:Data', NS) is not None:
                key = cells[0].find('ss:Data', NS).text
                if key == "Machine":
                    machine_id = cells[1].find('ss:Data', NS).text
                elif key == "COMPRESSOR NUMBER OF CYLINDERS":
                    num_cylinders = int(cells[2].find('ss:Data', NS).text)
        
        if num_cylinders == 0:
            st.warning("Could not determine number of cylinders from Source.xml. Defaulting to 1.")
            num_cylinders = 1

        curves_root = ET.fromstring(_curves_xml_content)
        ws_curves = next((ws for ws in curves_root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'), None)
        table = ws_curves.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        all_curve_names = [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]

        cylinders_config = []
        for i in range(1, num_cylinders + 1):
            pressure_curve = next((c for c in all_curve_names if f".{i}H." in c and "STATIC" in c), None)
            if not pressure_curve:
                pressure_curve = next((c for c in all_curve_names if f".{i}C." in c and "STATIC" in c), None)

            valve_curves = [
                {"name": "HE Discharge", "curve": next((c for c in all_curve_names if f".{i}HD" in c and "VIBRATION" in c), None)},
                {"name": "HE Suction", "curve": next((c for c in all_curve_names if f".{i}HS" in c and "VIBRATION" in c), None)},
                {"name": "CE Discharge", "curve": next((c for c in all_curve_names if f".{i}CD" in c and "VIBRATION" in c), None)},
                {"name": "CE Suction", "curve": next((c for c in all_curve_names if f".{i}CS" in c and "VIBRATION" in c), None)}
            ]
            
            if pressure_curve and any(vc['curve'] for vc in valve_curves):
                cylinders_config.append({
                    "cylinder_name": f"Cylinder {i}",
                    "pressure_curve": pressure_curve,
                    "valve_vibration_curves": [vc for vc in valve_curves if vc['curve']]
                })

        return {"machine_id": machine_id, "cylinders": cylinders_config}

    except Exception as e:
        st.error(f"An error occurred during auto-discovery: {e}")
        return None

# --- Core Diagnostics & Plotting ---

def generate_cylinder_view(df, cylinder_config, envelope_view, vertical_offset):
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

    fig, ax1 = plt.subplots(figsize=(16, 9))
    plt.style.use('seaborn-darkgrid')

    ax1.plot(df['Crank Angle'], df[pressure_curve], color='black', label='Pressure (PSI)', linewidth=2)
    ax1.fill_between(df['Crank Angle'], df[pressure_curve].min(), df[pressure_curve].max(), where=df[f'{pressure_curve}_anom'], color='gray', alpha=0.4, interpolate=True, label='Pressure Anomaly')
    ax1.set_ylabel('Pressure (PSI)', color='black', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel('Crank Angle (deg)', fontsize=14)

    ax2 = ax1.twinx()
    colors = plt.cm.viridis(np.linspace(0, 1, len(valve_curves)))
    
    current_offset = 0
    for i, vc in enumerate(valve_curves):
        curve_name = vc['curve']
        label_name = vc['name']
        vibration_data = df[curve_name] + current_offset
        
        if envelope_view:
            ax2.plot(df['Crank Angle'], vibration_data, color=colors[i], linewidth=0.5)
            ax2.plot(df['Crank Angle'], -df[curve_name] + current_offset, color=colors[i], linewidth=0.5)
            ax2.fill_between(df['Crank Angle'], vibration_data, -df[curve_name] + current_offset, color=colors[i], alpha=0.3, label=label_name, interpolate=True)
        else:
            ax2.plot(df['Crank Angle'], vibration_data, label=label_name, color=colors[i], linewidth=1.5)
            ax2.fill_between(df['Crank Angle'], vibration_data.min(), vibration_data.max(), where=df[f'{curve_name}_anom'], color=colors[i], alpha=0.3, interpolate=True)
        current_offset += vertical_offset

    ax2.set_ylabel('Vibration (G) with Offset', color='blue', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(False)

    fig.suptitle(f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}", fontsize=20, weight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, report_data

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")
st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")
st.markdown("Upload your machine's XML data files. The configuration will be discovered automatically.")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Data Files")
    uploaded_files = st.file_uploader(
        "Upload Machine XML Data (Curves, Levels, Source)", 
        type=["xml"], 
        accept_multiple_files=True
    )
    
    st.header("2. View Options")
    envelope_view = st.checkbox("Enable Envelope View", value=True)
    vertical_offset = st.slider("Vertical Offset", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    # --- Analysis History ---
    st.header("3. Analysis History")
    cursor = st.session_state.db_conn.cursor()
    sessions = cursor.execute("SELECT id, timestamp, machine_id FROM sessions ORDER BY timestamp DESC").fetchall()
    session_options = {f"{row[0]}: {row[2]} ({row[1]})": row[0] for row in sessions}
    selected_session_str = st.selectbox("Load a previous session", options=session_options.keys())


# --- Main Application Logic ---
if uploaded_files and len(uploaded_files) == 3:
    
    files_content = {}
    for file in uploaded_files:
        if 'curves' in file.name.lower():
            files_content['curves'] = file.getvalue().decode('utf-8')
        elif 'levels' in file.name.lower():
            files_content['levels'] = file.getvalue().decode('utf-8')
        elif 'source' in file.name.lower():
            files_content['source'] = file.getvalue().decode('utf-8')

    if 'curves' in files_content and 'levels' in files_content and 'source' in files_content:
        st.sidebar.success("All files uploaded successfully!")
        
        try:
            discovered_config = auto_discover_configuration(files_content['source'], files_content['curves'])
            df, all_curve_names = load_all_curves_data(files_content['curves'])
            
            # --- Save Session to DB ---
            rpm = extract_rpm(files_content['levels'])
            machine_id = discovered_config.get('machine_id', 'N/A')
            cursor = st.session_state.db_conn.cursor()
            cursor.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (machine_id, rpm))
            st.session_state.db_conn.commit()
            st.session_state.active_session_id = cursor.lastrowid
            st.info(f"New analysis session #{st.session_state.active_session_id} created.")


            if df is not None and discovered_config:
                cylinders = discovered_config.get("cylinders", [])
                if not cylinders:
                    st.error("Could not automatically discover any valid cylinder configurations.")
                else:
                    cylinder_names = [c.get("cylinder_name") for c in cylinders]
                    selected_cylinder_name = st.sidebar.selectbox("Select Cylinder", cylinder_names, key="cylinder_selector")
                    selected_cylinder_config = next((c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), None)

                    if selected_cylinder_config:
                        with st.spinner(f'Analyzing {selected_cylinder_name}...'):
                            fig, report_data = generate_cylinder_view(df.copy(), selected_cylinder_config, envelope_view, vertical_offset)
                        
                        st.header(f"📊 Diagnostic Chart for {selected_cylinder_name}")
                        st.pyplot(fig)
                        
                        st.header("📝 Diagnostic Summary")
                        st.markdown(f"**Machine ID:** {machine_id} | **Operating RPM:** {rpm}")
                        st.markdown(f"**Data Points Analyzed:** {len(df)}")
                        st.markdown("--- \n ### Anomaly Summary")
                        
                        analysis_ids = {}
                        for item in report_data:
                            st.markdown(f"- **{item['name']} Anomalies:** {item['count']} points (Threshold: {item['threshold']:.2f} {item['unit']})")
                            # Save analysis item to DB
                            cursor.execute(
                                "INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)",
                                (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold'])
                            )
                            analysis_ids[item['name']] = cursor.lastrowid
                        st.session_state.db_conn.commit()

                        # --- Anomaly Labeling UI ---
                        st.header("🏷️ Anomaly Labeling")
                        with st.expander("Add labels to detected anomalies"):
                            for item in report_data:
                                if item['count'] > 0:
                                    analysis_id = analysis_ids[item['name']]
                                    label_key = f"label_{analysis_id}"
                                    user_label = st.text_input(f"Label for {item['name']} anomaly:", key=f"txt_{label_key}")
                                    if st.button(f"Save Label for {item['name']}", key=f"btn_{label_key}"):
                                        if user_label:
                                            cursor.execute("INSERT INTO labels (analysis_id, label_text) VALUES (?, ?)", (analysis_id, user_label))
                                            st.session_state.db_conn.commit()
                                            st.success(f"Saved label for {item['name']}: '{user_label}'")
                                        else:
                                            st.warning("Please enter a label before saving.")
                        
                        # --- Display and Export ALL Labels from DB ---
                        all_labels = cursor.execute("SELECT s.timestamp, s.machine_id, a.cylinder_name, a.curve_name, l.label_text FROM labels l JOIN analyses a ON l.analysis_id = a.id JOIN sessions s ON a.session_id = s.id").fetchall()
                        if all_labels:
                            st.header("📋 All Saved Labels")
                            labels_df = pd.DataFrame(all_labels, columns=['Timestamp', 'Machine ID', 'Cylinder', 'Curve', 'Label'])
                            st.dataframe(labels_df)
                            
                            st.download_button(
                                label="Download All Labels as JSON",
                                data=json.dumps([dict(zip([column[0] for column in cursor.description], row)) for row in all_labels], indent=2),
                                file_name="all_anomaly_labels.json",
                                mime="application/json"
                            )

        except Exception as e:
            st.error(f"An error occurred during processing. Please check the files. Details: {e}")
    else:
        st.sidebar.error("Upload failed. Please ensure you upload one of each file type: 'Curves', 'Levels', and 'Source'.")
elif uploaded_files:
    st.sidebar.warning(f"Please upload all 3 required XML files. You have uploaded {len(uploaded_files)}.")
else:
    st.info("Please upload all three required XML files (Curves, Levels, Source) to begin the analysis.")

