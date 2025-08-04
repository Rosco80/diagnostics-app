"""
AI-Powered Machine Diagnostics Analyzer
========================================

Required Dependencies:
----------------------
Core dependencies (always required):
- streamlit
- pandas
- numpy
- matplotlib

Optional dependencies:
- reportlab (for PDF generation, fallback to HTML if not available)

Installation:
------------
pip install streamlit pandas numpy matplotlib
pip install reportlab  # Optional, for PDF generation

For Streamlit Cloud, create a requirements.txt file with:
streamlit
pandas
numpy
matplotlib
reportlab

Usage:
------
streamlit run app.py
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

# Optional PDF generation - handle missing reportlab gracefully
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be limited to HTML format. Install with: `pip install reportlab`")

# Standard fault labels for the dropdown menu
FAULT_LABELS = [
    "Valve Leakage",
    "Valve Wear",
    "Valve Sticking or Fouling",
    "Valve Impact or Slamming",
    "Broken or Missing Valve Parts",
    "Valve Misalignment",
    "Spring Fatigue or Failure",
    "Other"  # Allows for custom input
]


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
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = {}

# --- Helper Functions ---

def safe_db_operation(operation, *args):
    """Safely execute database operations with error handling"""
    try:
        cursor = db_conn.cursor()
        result = cursor.execute(operation, args)
        db_conn.commit()
        return result
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def validate_angle(angle, valve_name):
    """Validate crank angle input"""
    if angle is None:
        return True  # Allow None values
    
    if not (0 <= angle <= 720):  # Typical crank angle range
        st.warning(f"⚠️ {valve_name}: Crank angle should be between 0° and 720°")
        return False
    
    return True

def generate_pdf_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None):
    """Generate a PDF report of the analysis for a single cylinder."""
    if not REPORTLAB_AVAILABLE:
        return generate_html_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig)
    
    from reportlab.platypus import Image
    from reportlab.lib.units import inch
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"Machine Diagnostics Report - {machine_id}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Basic info
    info_text = f"<b>Machine ID:</b> {machine_id}<br/><b>RPM:</b> {rpm}<br/><b>Cylinder:</b> {cylinder_name}<br/><b>Analysis Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    info_para = Paragraph(info_text, styles['Normal'])
    story.append(info_para)
    story.append(Spacer(1, 12))
    
    # Add diagnostic chart if provided
    if chart_fig is not None:
        try:
            img_buffer = io.BytesIO()
            chart_fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            story.append(Paragraph("Diagnostic Chart", styles['h2']))
            story.append(Spacer(1, 6))
            story.append(Image(img_buffer, width=7*inch, height=4*inch))
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"<i>Chart could not be included: {str(e)}</i>", styles['Normal']))
    
    # Anomaly summary
    story.append(Paragraph("Anomaly Summary", styles['h2']))
    story.append(Spacer(1, 6))
    for item in report_data:
        status_icon = "●" if item['count'] > 0 else "○"
        anomaly_text = f"<b>{status_icon} {item['name']}:</b> {item['count']} anomalies detected (Threshold: {item['threshold']:.2f} {item['unit']})"
        story.append(Paragraph(anomaly_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Health report table
    if not health_report_df.empty:
        story.append(Paragraph("Health Report", styles['h2']))
        story.append(Spacer(1, 6))
        table_data = [health_report_df.columns.tolist()] + health_report_df.values.tolist()
        table = Table(table_data, colWidths=[0.8*inch, 0.9*inch, 1.1*inch, 1.5*inch, 1.2*inch, 0.8*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9), ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey), ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white]),
        ]))
        story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_html_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None):
    """Generate an HTML report as fallback when reportlab is not available"""
    
    chart_html = ""
    if chart_fig is not None:
        try:
            img_buffer = io.BytesIO()
            chart_fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            import base64
            chart_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            chart_html = f'<div class="chart-container"><img src="data:image/png;base64,{chart_b64}" style="max-width: 100%; height: auto;"></div>'
        except Exception as e:
            chart_html = f'<p style="color: #dc3545;"><i>Chart could not be included: {str(e)}</i></p>'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Machine Diagnostics Report - {machine_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
            .info-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .anomaly-item {{ margin: 10px 0; padding: 10px; background-color: #fff3cd; border-radius: 3px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px 8px; text-align: center; }}
            th {{ background-color: #343a40; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Machine Diagnostics Report - {machine_id}</h1>
        <div class="info-box">
            <strong>Machine ID:</strong> {machine_id}<br/>
            <strong>RPM:</strong> {rpm}<br/>
            <strong>Cylinder:</strong> {cylinder_name}<br/>
            <strong>Analysis Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        <h2>Diagnostic Chart</h2>
        {chart_html}
        <h2>Anomaly Summary</h2>
    """
    
    for item in report_data:
        status_class = "status-error" if item['count'] > 0 else "status-ok"
        status_icon = "●" if item['count'] > 0 else "○"
        html_content += f"""
        <div class="anomaly-item">
            <strong class="{status_class}">{status_icon} {item['name']}:</strong> 
            {item['count']} anomalies detected (Threshold: {item['threshold']:.2f} {item['unit']})
        </div>"""
    
    if not health_report_df.empty:
        html_content += "<h2>Health Report</h2>"
        html_content += health_report_df.to_html(index=False, border=0)
    
    html_content += "</body></html>"
    return io.BytesIO(html_content.encode('utf-8'))

@st.cache_data
def load_all_curves_data(_curves_xml_content):
    """
    Parses the entire Curves.xml file once and caches the resulting DataFrame.
    Returns the DataFrame and the list of column names actually used, which prevents KeyErrors.
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
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]

        data = []
        for r in rows[6:]:
            cells = r.findall('ss:Cell', NS)
            row_data = [cell.find('ss:Data', NS).text for cell in cells]
            data.append(row_data)
        
        if not data:
            st.error("No data found in 'Curves' worksheet.")
            return None, None
            
        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]
        
        df = pd.DataFrame(data, columns=actual_columns)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)

        return df, actual_columns
    except Exception as e:
        st.error(f"Failed to load or parse curves data: {e}")
        return None, None


def extract_rpm(_levels_xml_content):
    """Extract machine RPM from the Levels.xml file."""
    try:
        root = ET.fromstring(_levels_xml_content)
        rpm_str = find_xml_value(root, 'Levels', 'RPM', 1)
        if rpm_str != "N/A":
            return f"{float(rpm_str):.0f}"
    except Exception:
        return "N/A"
    return "N/A"

@st.cache_data
def auto_discover_configuration(_source_xml_content, all_curve_names):
    """
    Automatically discovers the machine configuration from the Source.xml file
    and the provided list of available curve names.
    """
    try:
        source_root = ET.fromstring(_source_xml_content)
        
        num_cyl_str = find_xml_value(source_root, 'Source', "COMPRESSOR NUMBER OF CYLINDERS", 2)
        machine_id = find_xml_value(source_root, 'Source', "Machine", 1)

        if num_cyl_str == "N/A" or int(num_cyl_str) == 0:
            st.warning("Could not determine number of cylinders from Source.xml.")
            return None
        
        num_cylinders = int(num_cyl_str)
        
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

def find_xml_value(root, sheet_name, partial_key, col_offset, cylinder_id_pattern=None, occurrence=1):
    """
    Robustly finds a value in a worksheet.
    - If cylinder_id_pattern is None, it uses partial_key and col_offset.
    - If cylinder_id_pattern is provided, it finds the row with partial_key and then
      searches that row for a cell containing the cylinder_id_pattern.
    - `occurrence` specifies which match to use if multiple rows contain the partial_key (1-based).
    """
    try:
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
        if ws is None: 
            return "N/A"
        
        rows = ws.findall('.//ss:Row', NS)
        match_count = 0
        for row in rows:
            all_cells_in_row = row.findall('ss:Cell', NS)
            if not all_cells_in_row:
                continue
            
            first_cell_data_node = all_cells_in_row[0].find('ss:Data', NS)
            if first_cell_data_node is None or first_cell_data_node.text is None:
                continue

            cell_text = (first_cell_data_node.text or "").strip().upper()
            if partial_key.upper() not in cell_text:
                continue

            match_count += 1
            if match_count != occurrence:
                continue

            # --- LOGIC FOR DIFFERENT DATA STRUCTURES ---
            
            # Case 1: Data is identified by a cylinder-specific pattern within the row
            if cylinder_id_pattern:
                for cell in all_cells_in_row:
                    data_node = cell.find('ss:Data', NS)
                    if data_node is not None and data_node.text:
                        if cylinder_id_pattern in data_node.text:
                            # The next cell usually contains the value
                            try:
                                value_cell_index = all_cells_in_row.index(cell) + 1
                                value_node = all_cells_in_row[value_cell_index].find('ss:Data', NS)
                                return value_node.text if value_node is not None else "N/A"
                            except (IndexError, AttributeError):
                                return "N/A" # No next cell
                return "N/A" # Pattern not found in any cell of the matched row

            # Case 2: Data is in a fixed column offset (handles ss:Index)
            else:
                dense_cells = {}
                current_idx = 1 # Spreadsheet columns are 1-based
                for cell in all_cells_in_row:
                    ss_index_str = cell.get(f'{{{NS["ss"]}}}Index')
                    if ss_index_str:
                        current_idx = int(ss_index_str)
                    dense_cells[current_idx] = cell
                    current_idx += 1
                
                # Adjust col_offset to be 1-based for lookup
                target_idx = col_offset + 1
                if target_idx in dense_cells:
                    value_node = dense_cells[target_idx].find('ss:Data', NS)
                    return value_node.text if value_node is not None and value_node.text else "N/A"
                else:
                    return "N/A"

        return "N/A"
    except Exception:
        return "N/A"


def generate_health_report_table(_source_xml_content, _levels_xml_content, cylinder_index):
    """Generates a DataFrame for the health report table using robust parsing."""
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        
        col_idx = cylinder_index + 1

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str: return "N/A"
            try:
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str
        
        # Fetch stage-level data from Levels.xml (value is in the 3rd cell, so offset is 2)
        suction_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2))
        discharge_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2))
        suction_temp = find_xml_value(levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2)
        
        # Fetch per-cylinder discharge temp from Levels.xml
        discharge_temp = find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx)

        # Fetch per-cylinder data from Source.xml
        bore = find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx)
        rod_diam = find_xml_value(source_root, 'Source', 'PISTON ROD DIAMETER', col_idx)
        
        # Use occurrence to get the correct values from repeated rows
        comp_ratio_he = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx, occurrence=2) # HEAD END is second match
        comp_ratio_ce = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx, occurrence=1) # CRANK END is first match
        
        power_he = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx, occurrence=2)
        power_ce = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx, occurrence=1)

        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'],
            'Bore (ins)': [bore] * 2,
            'Rod Diam (ins)': ['N/A', rod_diam],
            'Pressure Ps/Pd (psig)': [f"{suction_p} / {discharge_p}"] * 2,
            'Temp Ts/Td (°C)': [f"{suction_temp} / {discharge_temp}"] * 2,
            'Comp. Ratio': [comp_ratio_he, comp_ratio_ce],
            'Indicated Power (ihp)': [power_he, power_ce]
        }
        
        df_table = pd.DataFrame(data)
        return df_table

    except Exception as e:
        st.warning(f"Could not generate health report table: {e}")
        return pd.DataFrame()

def get_all_cylinder_details(_source_xml_content, _levels_xml_content, num_cylinders):
    """Extracts key details for all cylinders for the summary cards."""
    details = []
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str: return "N/A"
            try:
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str

        def format_flow_balance(value_str):
            """Converts flow balance ratio string to a formatted percentage string."""
            if value_str == "N/A" or not value_str:
                return "N/A"
            try:
                # The value from XML is a ratio (e.g., 1.015). Convert to percentage.
                percent_val = float(value_str) * 100
                return f"{percent_val:.1f} %"
            except (ValueError, TypeError):
                # If it's not a number, return it as is
                return value_str

        # --- Fetch Stage-Level Data (single value for all cylinders) ---
        # Note: col_offset=2 because value is in the 3rd cell (index 2).
        stage_suction_p_psi = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2))
        stage_discharge_p_psi = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2))
        stage_suction_temp = find_xml_value(levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2)

        for i in range(1, num_cylinders + 1):
            col_idx = i + 1
            
            # Fetch flow balance values first
            fb_ce_raw = find_xml_value(source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=1)
            fb_he_raw = find_xml_value(source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=2)
            
            detail = {
                "name": f"Cylinder {i}",
                "bore": f"{find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx)} in",
                "suction_temp": f"{stage_suction_temp} °C",
                "discharge_temp": f"{find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx)} °C",
                "suction_pressure": f"{stage_suction_p_psi} psig",
                "discharge_pressure": f"{stage_discharge_p_psi} psig",
                "flow_balance_ce": format_flow_balance(fb_ce_raw),
                "flow_balance_he": format_flow_balance(fb_he_raw)
            }

            for key, value in detail.items():
                if "N/A" in str(value) or not str(value).strip():
                    detail[key] = "N/A"
            
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

    # --- Plotly Implementation ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Pressure Trace
    fig.add_trace(
        go.Scatter(x=df['Crank Angle'], y=df[pressure_curve], name='Pressure (PSI)',
                   line=dict(color='black', width=2), hovertemplate='Angle: %{x:.2f}°<br>Pressure: %{y:.2f} PSI'),
        secondary_y=False,
    )
    # Add Pressure Anomaly Shading
    anomaly_pressure = df[pressure_curve].where(df[f'{pressure_curve}_anom'])
    fig.add_trace(
        go.Scatter(x=df['Crank Angle'], y=anomaly_pressure, name='Pressure Anomaly',
                   mode='lines', line=dict(width=0), showlegend=False,
                   fillcolor='rgba(255, 0, 0, 0.3)', fill='tozeroy',
                   hovertemplate='Anomaly: %{y:.2f} PSI'),
        secondary_y=False,
    )

    # Add Vibration Traces and Valve Events
    colors = plt.cm.viridis(np.linspace(0, 1, len(valve_curves)))
    cursor = db_conn.cursor()
    current_offset = 0

    for i, vc in enumerate(valve_curves):
        curve_name = vc['curve']
        label_name = vc['name']
        color_rgba = f'rgba({colors[i][0]*255},{colors[i][1]*255},{colors[i][2]*255},0.4)'

        # Add Vibration Envelope
        if envelope_view:
            upper_bound = df[curve_name] + current_offset
            lower_bound = -df[curve_name] + current_offset
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=upper_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), showlegend=False, hoverinfo='none'), secondary_y=True)
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=lower_bound, mode='lines', line=dict(width=0.5, color=color_rgba.replace('0.4','1')), fill='tonexty', fillcolor=color_rgba, name=label_name, hoverinfo='none'), secondary_y=True)
        else:
            vibration_data = df[curve_name] + current_offset
            fig.add_trace(go.Scatter(x=df['Crank Angle'], y=vibration_data, name=label_name, line=dict(color=color_rgba.replace('0.4','1'))), secondary_y=True)

        # Get and plot valve events
        analysis_id = analysis_ids.get(vc['name'])
        if analysis_id:
            events_raw = cursor.execute("SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ?", (analysis_id,)).fetchall()
            events = {etype: angle for etype, angle in events_raw}

            # Add Duration Shading
            if 'open' in events and 'close' in events:
                fig.add_vrect(x0=events['open'], x1=events['close'], fillcolor=color_rgba, layer="below", line_width=0, annotation_text=f"{label_name} Open", annotation_position="top left")

            # Add Markers
            for event_type, crank_angle in events.items():
                fig.add_vline(x=crank_angle, line_width=2, line_dash="dash", line_color='green' if event_type == 'open' else 'red')
        
        current_offset += vertical_offset
    
    # Update layout
    fig.update_layout(
        title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}",
        xaxis_title="Crank Angle (deg)",
        template="ggplot2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Pressure (PSI)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", secondary_y=True)
    
    return fig, report_data

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")

st.markdown("""
<style>
.detail-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    height: 100%;
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
        st.session_state.active_session_id = None
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
            df = None
            discovered_config = None
            with st.spinner("Analyzing machine data..."):
                df, actual_curve_names = load_all_curves_data(files_content['curves'])
                if df is not None and actual_curve_names is not None:
                    discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
            
            if df is not None and discovered_config:
                rpm = extract_rpm(files_content['levels'])
                machine_id = discovered_config.get('machine_id', 'N/A')
                
                if st.session_state.active_session_id is None:
                    cursor = db_conn.cursor()
                    cursor.execute("INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)", (machine_id, rpm))
                    db_conn.commit()
                    st.session_state.active_session_id = cursor.lastrowid
                    st.success(f"✅ New analysis session #{st.session_state.active_session_id} created.")

                cylinders = discovered_config.get("cylinders", [])
                if not cylinders:
                    st.error("Could not automatically discover any valid cylinder configurations.")
                else:
                    cylinder_names = [c.get("cylinder_name") for c in cylinders]
                    selected_cylinder_name = st.sidebar.selectbox("Select Cylinder for Detailed View", cylinder_names, key="cylinder_selector")
                    
                    selected_cylinder_config = next((c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name), None)

                    if selected_cylinder_config:
                        _, temp_report_data = generate_cylinder_view(df.copy(), selected_cylinder_config, envelope_view, vertical_offset, {})
                        
                        analysis_ids = {}
                        cursor = db_conn.cursor()
                        for item in temp_report_data:
                            # Check if an analysis record already exists for this curve in this session/cylinder
                            cursor.execute("""
                                SELECT id FROM analyses 
                                WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?
                            """, (st.session_state.active_session_id, selected_cylinder_name, item['curve_name']))
                            
                            existing_analysis = cursor.fetchone()

                            if existing_analysis:
                                # If it exists, reuse the ID and update the values
                                analysis_id = existing_analysis[0]
                                cursor.execute("UPDATE analyses SET anomaly_count = ?, threshold = ? WHERE id = ?",
                                    (item['count'], item['threshold'], analysis_id))
                            else:
                                # If not, insert a new record and get its ID
                                cursor.execute("""
                                    INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) 
                                    VALUES (?, ?, ?, ?, ?)
                                """, (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold']))
                                analysis_id = cursor.lastrowid
                            
                            analysis_ids[item['name']] = analysis_id
                        db_conn.commit()

                        fig, report_data = generate_cylinder_view(df.copy(), selected_cylinder_config, envelope_view, vertical_offset, analysis_ids)
                        
                        st.header(f"📊 Diagnostic Chart for {selected_cylinder_name}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("📋 Compressor Health Report")
                        cylinder_index = int(re.search(r'\d+', selected_cylinder_name).group())
                        health_report_df = generate_health_report_table(files_content['source'], files_content['levels'], cylinder_index)
                        if not health_report_df.empty:
                            st.dataframe(health_report_df, use_container_width=True, hide_index=True)

                        with st.expander("Add labels and mark valve events"):
                            st.subheader("Fault Labels")
                            
                            for item in report_data:
                                if item['count'] > 0:
                                    analysis_id = analysis_ids[item['name']]
                                    with st.form(key=f"label_form_{analysis_id}"):
                                        st.write(f"**{item['name']} Anomaly** ({item['count']} points detected)")
                                        
                                        # Use a selectbox for standard labels
                                        selected_label = st.selectbox(
                                            "Select fault label:",
                                            options=FAULT_LABELS, # References the list from the top
                                            key=f"sel_label_{analysis_id}"
                                        )
                                        
                                        # Show a text input only if "Other" is selected
                                        custom_label = ""
                                        if selected_label == "Other":
                                            custom_label = st.text_input(
                                                "Enter custom label:",
                                                key=f"txt_label_{analysis_id}"
                                            )
                                        
                                        submitted = st.form_submit_button("Save Label")
                                        if submitted:
                                            # Determine the final label to save
                                            final_label = custom_label if selected_label == "Other" else selected_label
                                            
                                            if final_label.strip():
                                                safe_db_operation("INSERT INTO labels (analysis_id, label_text) VALUES (?, ?)", analysis_id, final_label.strip())
                                                st.success(f"✅ Label saved for {item['name']}: '{final_label}'")
                                            else:
                                                st.warning("⚠️ Please select or enter a label before saving.")

                            st.subheader("Mark Valve Open/Close Events")
                            for item in report_data:
                                if item['name'] != 'Pressure':  # Only for valves
                                    analysis_id = analysis_ids[item['name']]
                                    with st.form(key=f"valve_form_{analysis_id}"):
                                        st.write(f"**{item['name']} Valve Events:**")
                                        cols = st.columns(2)
                                        open_angle = cols[0].number_input("Open Angle", key=f"open_{analysis_id}", value=None, format="%.2f", help="Enter the crank angle where the valve opens")
                                        close_angle = cols[1].number_input("Close Angle", key=f"close_{analysis_id}", value=None, format="%.2f", help="Enter the crank angle where the valve closes")
                                        submitted = st.form_submit_button(f"Save Events for {item['name']}")
                                        if submitted:
                                            safe_db_operation("DELETE FROM valve_events WHERE analysis_id = ?", analysis_id)
                                            if open_angle is not None:
                                                safe_db_operation("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", analysis_id, 'open', open_angle)
                                            if close_angle is not None:
                                                safe_db_operation("INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)", analysis_id, 'close', close_angle)
                                            st.success(f"✅ Events updated for {item['name']}.")
                                            st.rerun() # Rerun to show the updated plot immediately

                        st.header("📄 Export Report")
                        if st.button("🔄 Generate Report for this Cylinder", type="primary"):
                            with st.spinner("Generating report..."):
                                report_buffer = generate_pdf_report(machine_id, rpm, selected_cylinder_name, report_data, health_report_df, fig)
                                file_ext = "pdf" if REPORTLAB_AVAILABLE else "html"
                                st.download_button(
                                    label=f"📥 Download {file_ext.upper()} Report",
                                    data=report_buffer.getvalue(),
                                    file_name=f"report_{machine_id}_{selected_cylinder_name}.{file_ext}",
                                    mime="application/pdf" if REPORTLAB_AVAILABLE else "text/html"
                                )

                        st.markdown("---")
                        st.header("🔧 All Cylinder Details")
                        all_details = get_all_cylinder_details(files_content['source'], files_content['levels'], len(cylinders))
                        if all_details:
                            cols = st.columns(len(all_details) or 1)
                            for i, detail in enumerate(all_details):
                                with cols[i]:
                                    st.markdown(f"""
                                    <div class="detail-card">
                                        <h5>{detail['name']}</h5>
                                        <div class="detail-item"><span>Bore:</span> <strong>{detail['bore']}</strong></div>
                                        <div class="detail-item"><span>Suction Temp:</span> <strong>{detail['suction_temp']}</strong></div>
                                        <div class="detail-item"><span>Discharge Temp:</span> <strong>{detail['discharge_temp']}</strong></div>
                                        <div class="detail-item"><span>Suction Pressure:</span> <strong>{detail['suction_pressure']}</strong></div>
                                        <div class="detail-item"><span>Discharge Pressure:</span> <strong>{detail['discharge_pressure']}</strong></div>
                                        <div class="detail-item"><span>Flow Balance (CE):</span> <strong>{detail['flow_balance_ce']}</strong></div>
                                        <div class="detail-item"><span>Flow Balance (HE):</span> <strong>{detail['flow_balance_he']}</strong></div>
                                    </div>
                                    """, unsafe_allow_html=True)

            elif df is None:
                st.error("Failed to load data from Curves.xml. Cannot proceed.")
            else:
                st.error("Could not discover a valid machine configuration from the provided files.")

        except Exception as e:
            st.error(f"❌ An error occurred during processing. Please check the files. Details: {e}")
            st.exception(e)
    else:
        st.sidebar.error("❌ Upload failed. Please ensure you upload one of each file type: 'Curves', 'Levels', and 'Source'.")
elif uploaded_files:
    st.sidebar.warning(f"⚠️ Please upload all 3 required XML files. You have uploaded {len(uploaded_files)}.")
else:
    st.info("📁 Please upload all three required XML files (Curves, Levels, Source) to begin the analysis.")

# --- Display All Saved Labels ---
st.header("📋 All Saved Labels")
st.markdown("Historical data from all analysis sessions.")

try:
    cursor = db_conn.cursor()
    query = """
    SELECT s.timestamp, s.machine_id, a.cylinder_name, a.curve_name, l.label_text 
    FROM labels l 
    JOIN analyses a ON l.analysis_id = a.id 
    JOIN sessions s ON a.session_id = s.id
    """
    params = []
    if selected_machine_id_filter != "All":
        query += " WHERE s.machine_id = ?"
        params.append(selected_machine_id_filter)
    query += " ORDER BY s.timestamp DESC"

    all_labels = cursor.execute(query, params).fetchall()

    if all_labels:
        labels_df = pd.DataFrame(all_labels, columns=['Timestamp', 'Machine ID', 'Cylinder', 'Curve', 'Label'])
        st.dataframe(labels_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        csv_data = labels_df.to_csv(index=False).encode('utf-8')
        col1.download_button("📊 Download Labels as CSV", csv_data, "anomaly_labels.csv", "text/csv")
        
        json_data = labels_df.to_json(orient='records', indent=2).encode('utf-8')
        col2.download_button("📋 Download Labels as JSON", json_data, "anomaly_labels.json", "application/json")
        
    else:
        st.info("📝 No labels found. Start analyzing data and adding labels to build your diagnostic knowledge base.")

except Exception as e:
    st.error(f"❌ Error retrieving labels: {str(e)}")

# --- Footer ---
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🔧 AI-Powered Machine Diagnostics Analyzer | 
    Session ID: {st.session_state.active_session_id or "None"} | 
    Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)


