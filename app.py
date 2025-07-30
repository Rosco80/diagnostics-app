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
    # New table for valve open/close events
    c.execute('''
        CREATE TABLE IF NOT EXISTS valve_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            event_type TEXT, -- "open" or "close"
            crank_angle REAL,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    conn.commit()
    return conn

# --- Session State Initialization ---
db_conn = init_db()
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'print_report' not in st.session_state:
    st.session_state.print_report = False


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

def extract_temperature(_levels_xml_content, cylinder_index):
    """Extracts discharge temperature for a specific cylinder index."""
    try:
        root = ET.fromstring(_levels_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws_levels = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Levels'), None)
        if ws_levels is None: return "N/A"
        
        table = ws_levels.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        for row in rows:
            cells = row.findall('ss:Cell', NS)
            if len(cells) > cylinder_index and cells[0].find('ss:Data', NS) is not None:
                cell_text = cells[0].find('ss:Data', NS).text
                if cell_text and "DISCHARGE TEMPERATURE" in cell_text:
                    temp_node = cells[cylinder_index].find('ss:Data', NS)
                    if temp_node is not None and temp_node.text:
                        return f"{float(temp_node.text):.1f}°C"
    except (ValueError, IndexError):
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

# --- New Health Report Table Function ---
def generate_health_report_table(_source_xml_content, _levels_xml_content, cylinder_index):
    """Generates a DataFrame for the health report table."""
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

        def find_value(root, sheet_name, key_name, col_offset, is_source=False):
            ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
            if ws is None: return "N/A"
            rows = ws.findall('.//ss:Row', NS)
            for row in rows:
                cells = row.findall('ss:Cell', NS)
                # Source.xml has a different structure (key, unit, value) vs Levels.xml (key, unit, val1, val2...)
                data_col = col_offset if is_source else col_offset + 1
                if len(cells) > data_col and cells[0].find('ss:Data', NS) is not None:
                    cell_text = cells[0].find('ss:Data', NS).text
                    if cell_text and key_name in cell_text:
                        value_node = cells[data_col].find('ss:Data', NS)
                        return value_node.text if value_node is not None and value_node.text else "N/A"
            return "N/A"

        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'],
            'Bore (ins)': [find_value(source_root, 'Source', f'CYLINDER {cylinder_index} BORE DIAMETER', 2, is_source=True)] * 2,
            'Rod Diam (ins)': ['N/A', find_value(source_root, 'Source', f'CYLINDER {cylinder_index} PISTON ROD DIAMETER', 2, is_source=True)],
            'Pressure Ps/Pd (psig)': [
                f"{find_value(levels_root, 'Levels', 'SUCTION PRESSURE', cylinder_index)} / {find_value(levels_root, 'Levels', 'DISCHARGE PRESSURE', cylinder_index)}",
                f"{find_value(levels_root, 'Levels', 'SUCTION PRESSURE', cylinder_index)} / {find_value(levels_root, 'Levels', 'DISCHARGE PRESSURE', cylinder_index)}"
            ],
            'Temp Ts/Td': [
                f"{find_value(levels_root, 'Levels', 'SUCTION TEMPERATURE', cylinder_index)} / {find_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', cylinder_index)}",
                f"{find_value(levels_root, 'Levels', 'SUCTION TEMPERATURE', cylinder_index)} / {find_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', cylinder_index)}"
            ],
            'Comp. Ratio': [find_value(levels_root, 'Levels', 'COMPRESSION RATIO', cylinder_index)] * 2,
            'Indicated Power (ihp)': [
                find_value(levels_root, 'Levels', 'HEAD END INDICATED HORSEPOWER', cylinder_index),
                find_value(levels_root, 'Levels', 'CRANK END INDICATED HORSEPOWER', cylinder_index)
            ]
        }
        
        df_table = pd.DataFrame(data)
        return df_table

    except Exception as e:
        st.warning(f"Could not generate health report table: {e}")
        return pd.DataFrame()

# --- New Cylinder Details Card Function ---
def get_all_cylinder_details(_source_xml_content, _levels_xml_content, num_cylinders):
    """Extracts key details for all cylinders for the summary cards."""
    details = []
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

        def find_value(root, sheet_name, key_name, col_offset, is_source=False):
            ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
            if ws is None: return "N/A"
            rows = ws.findall('.//ss:Row', NS)
            for row in rows:
                cells = row.findall('ss:Cell', NS)
                data_col = col_offset if is_source else col_offset + 1
                if len(cells) > data_col and cells[0].find('ss:Data', NS) is not None:
                    cell_text = cells[0].find('ss:Data', NS).text
                    if cell_text and key_name in cell_text:
                        value_node = cells[data_col].find('ss:Data', NS)
                        return value_node.text if value_node is not None and value_node.text else "N/A"
            return "N/A"

        for i in range(1, num_cylinders + 1):
            detail = {
                "name": f"Cylinder {i}",
                "bore": find_value(source_root, 'Source', f'CYLINDER {i} BORE DIAMETER', 2, is_source=True),
                "suction_temp": find_value(levels_root, 'Levels', 'SUCTION TEMPERATURE', i),
                "discharge_temp": find_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', i),
                "suction_pressure": find_value(levels_root, 'Levels', 'SUCTION PRESSURE', i),
                "discharge_pressure": find_value(levels_root, 'Levels', 'DISCHARGE PRESSURE', i),
                "flow_balance_ce": find_value(levels_root, 'Levels', 'CRANK END FLOW BALANCE', i),
                "flow_balance_he": find_value(levels_root, 'Levels', 'HEAD END FLOW BALANCE', i)
            }
            details.append(detail)
        return details
    except Exception as e:
        st.warning(f"Could not extract all cylinder details: {e}")
        return []


# --- Core Diagnostics & Plotting ---

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

    fig, ax1 = plt.subplots(figsize=(16, 9))
    plt.style.use('ggplot')

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

    cursor = db_conn.cursor()
    for item in report_data:
        if item['name'] != 'Pressure':
            analysis_id = analysis_ids.get(item['name'])
            if analysis_id:
                events = cursor.execute("SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?", (analysis_id,)).fetchall()
                for event_type, crank_angle in events:
                    if event_type == 'open':
                        ax1.axvline(x=crank_angle, color='g', linestyle='--', linewidth=2)
                        ax1.text(crank_angle + 2, ax1.get_ylim()[1]*0.9, 'O', color='g', fontsize=12, weight='bold')
                    elif event_type == 'close':
                        ax1.axvline(x=crank_angle, color='r', linestyle='--', linewidth=2)
                        ax1.text(crank_angle + 2, ax1.get_ylim()[1]*0.9, 'C', color='r', fontsize=12, weight='bold')

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

st.markdown("""
<style>
@media print {
    .stSidebar, .stToolbar, .stActionButton {
        display: none !important;
    }
    .main .block-container {
        padding: 1rem !important;
    }
}
.detail-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
}
.detail-card h5 {
    color: #007bff;
    margin-bottom: 1rem;
}
.detail-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}
.detail-item span:first-child {
    color: #6c757d;
}
</style>
""", unsafe_allow_html=True)


st.title("⚙️ AI-Powered Machine Diagnostics Analyzer")
st.markdown("Upload your machine's XML data files. The configuration will be discovered automatically.")

with st.sidebar:
    st.header("1. Upload Data Files")
    uploaded_files = st.file_uploader(
        "Upload Machine XML Data (Curves, Levels, Source)", 
        type=["xml"], 
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    if st.button("Start New Analysis / Clear Files"):
        st.session_state.file_uploader_key += 1
        st.rerun()

    st.header("2. View Options")
    envelope_view = st.checkbox("Enable Envelope View", value=True)
    vertical_offset = st.slider("Vertical Offset", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    st.header("3. View All Saved Labels")
    st.caption("Shows all labels saved from all past analysis sessions.")
    cursor = db_conn.cursor()
    machine_ids = cursor.execute("SELECT DISTINCT machine_id FROM sessions ORDER BY machine_id ASC").fetchall()
    machine_id_options = [row[0] for row in machine_ids]
    selected_machine_id_filter = st.selectbox("Filter labels by Machine ID", options=["All"] + machine_id_options)
    
    st.header("4. Export")
    if st.button("Print Report (PDF)"):
        st.session_state.print_report = True


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
            
            rpm = extract_rpm(files_content['levels'])
            machine_id = discovered_config.get('machine_id', 'N/A')
            cursor = db_conn.cursor()
            cursor.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (machine_id, rpm))
            db_conn.commit()
            st.session_state.active_session_id = cursor.lastrowid
            st.info(f"New analysis session #{st.session_state.active_session_id} created.")

            if df is not None and discovered_config:
                cylinders = discovered_config.get("cylinders", [])
                if not cylinders:
                    st.error("Could not automatically discover any valid cylinder configurations.")
                else:
                    cylinder_names = [c.get("cylinder_name") for c in cylinders]
                    selected_cylinder_name = st.sidebar.selectbox("Select Cylinder for Detailed View", cylinder_names, key="cylinder_selector")
                    
                    selected_cylinder_config = next((c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), None)

                    if selected_cylinder_config:
                        # Perform analysis once to get report data
                        _, temp_report_data = generate_cylinder_view(df.copy(), selected_cylinder_config, envelope_view, vertical_offset, {})
                        
                        analysis_ids = {}
                        for item in temp_report_data:
                            cursor.execute( "INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)",
                                (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold']))
                            analysis_ids[item['name']] = cursor.lastrowid
                        db_conn.commit()

                        fig, report_data = generate_cylinder_view(df.copy(), selected_cylinder_config, envelope_view, vertical_offset, analysis_ids)
                        
                        st.header(f"📊 Diagnostic Chart for {selected_cylinder_name}")
                        st.pyplot(fig)
                        
                        st.header("🏷️ Anomaly Labeling")
                        with st.expander("Add labels to detected anomalies and valve events"):
                            st.subheader("Fault Labels")
                            for item in report_data:
                                if item['count'] > 0:
                                    analysis_id = analysis_ids[item['name']]
                                    label_key = f"label_{analysis_id}"
                                    user_label = st.text_input(f"Label for {item['name']} anomaly:", key=f"txt_{label_key}")
                                    if st.button(f"Save Label for {item['name']}", key=f"btn_{label_key}"):
                                        if user_label:
                                            cursor.execute("INSERT INTO labels (analysis_id, label_text) VALUES (?, ?)", (analysis_id, user_label))
                                            db_conn.commit()
                                            st.success(f"Saved label for {item['name']}: '{user_label}'")
                                        else:
                                            st.warning("Please enter a label before saving.")
                            
                            st.subheader("Mark Valve Open/Close Events")
                            for item in report_data:
                                if item['name'] != 'Pressure': # Only for valves
                                    analysis_id = analysis_ids[item['name']]
                                    cols = st.columns([3, 2, 2, 2])
                                    with cols[0]:
                                        st.write(f"**{item['name']}:**")
                                    with cols[1]:
                                        open_angle = st.number_input("Open Angle", key=f"open_{analysis_id}", value=None, format="%f")
                                    with cols[2]:
                                        close_angle = st.number_input("Close Angle", key=f"close_{analysis_id}", value=None, format="%f")
                                    with cols[3]:
                                        st.write("") 
                                        st.write("") 
                                        if st.button("Save Events", key=f"btn_event_{analysis_id}"):
                                            cursor.execute("DELETE FROM valve_events WHERE analysis_id = ?", (analysis_id,))
                                            if open_angle is not None:
                                                cursor.execute("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", (analysis_id, 'open', open_angle))
                                            if close_angle is not None:
                                                cursor.execute("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", (analysis_id, 'close', close_angle))
                                            db_conn.commit()
                                            st.success(f"Events for {item['name']} saved.")
                                            st.rerun()

                        st.header("📝 Diagnostic Summary")
                        cylinder_index = int(re.search(r'\d+', selected_cylinder_name).group())
                        discharge_temp = extract_temperature(files_content['levels'], cylinder_index)
                        st.markdown(f"**Machine ID:** {machine_id} | **Operating RPM:** {rpm} | **Discharge Temp:** {discharge_temp}")
                        st.markdown(f"**Data Points Analyzed:** {len(df)}")
                        st.markdown("--- \n ### Anomaly Summary")
                        for item in report_data:
                            st.markdown(f"- **{item['name']} Anomalies:** {item['count']} points (Threshold: {item['threshold']:.2f} {item['unit']})")

                        st.header("Compressor Health Report")
                        health_report_df = generate_health_report_table(files_content['source'], files_content['levels'], cylinder_index)
                        if not health_report_df.empty:
                            st.dataframe(health_report_df)
                        
                        st.header("Cylinder Details")
                        all_details = get_all_cylinder_details(files_content['source'], files_content['levels'], len(cylinders))
                        cols = st.columns(len(all_details) if all_details else 1)
                        for i, detail in enumerate(all_details):
                            with cols[i]:
                                st.markdown(f"""
                                <div class="detail-card">
                                    <h5>{detail['name']}</h5>
                                    <div class="detail-item"><span>BORE:</span> <strong>{detail['bore']}</strong></div>
                                    <div class="detail-item"><span>Suction Temp:</span> <strong>{detail['suction_temp']}</strong></div>
                                    <div class="detail-item"><span>Discharge Temp:</span> <strong>{detail['discharge_temp']}</strong></div>
                                    <div class="detail-item"><span>Suction Pressure:</span> <strong>{detail['suction_pressure']}</strong></div>
                                    <div class="detail-item"><span>Discharge Pressure:</span> <strong>{detail['discharge_pressure']}</strong></div>
                                    <div class="detail-item"><span>Flow Balance (CE):</span> <strong>{detail['flow_balance_ce']}</strong></div>
                                    <div class="detail-item"><span>Flow Balance (HE):</span> <strong>{detail['flow_balance_he']}</strong></div>
                                </div>
                                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during processing. Please check the files. Details: {e}")
    else:
        st.sidebar.error("Upload failed. Please ensure you upload one of each file type: 'Curves', 'Levels', and 'Source'.")
elif uploaded_files:
    st.sidebar.warning(f"Please upload all 3 required XML files. You have uploaded {len(uploaded_files)}.")
else:
    st.info("Please upload all three required XML files (Curves, Levels, Source) to begin the analysis.")

# --- Display All Saved Labels ---
st.header("📋 All Saved Labels")
query = "SELECT s.timestamp, s.machine_id, a.cylinder_name, a.curve_name, l.label_text FROM labels l JOIN analyses a ON l.analysis_id = a.id JOIN sessions s ON a.session_id = s.id"
params = []
if selected_machine_id_filter != "All":
    query += " WHERE s.machine_id = ?"
    params.append(selected_machine_id_filter)
query += " ORDER BY s.timestamp DESC"

all_labels = cursor.execute(query, params).fetchall()

if all_labels:
    labels_df = pd.DataFrame(all_labels, columns=['Timestamp', 'Machine ID', 'Cylinder', 'Curve', 'Label'])
    st.dataframe(labels_df)
    
    st.download_button(
        label="Download All Labels as JSON",
        data=json.dumps([dict(zip([column[0] for column in cursor.description], row)) for row in all_labels], indent=2),
        file_name="all_anomaly_labels.json",
        mime="application/json"
    )

# --- Trigger Print Action ---
if st.session_state.print_report:
    st.components.v1.html('<script>window.print()</script>')
    st.session_state.print_report = False
