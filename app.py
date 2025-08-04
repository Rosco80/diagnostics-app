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

# Optional PDF generation - handle missing reportlab gracefully
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be limited to HTML format. Install with: `pip install reportlab`")

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
    """Generate a PDF report of the analysis"""
    if not REPORTLAB_AVAILABLE:
        return generate_html_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig)
    
    from reportlab.platypus import Image, PageBreak
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
            # Save matplotlib figure to bytes
            img_buffer = io.BytesIO()
            chart_fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Add chart to PDF
            chart_title = Paragraph("Diagnostic Chart", styles['Heading2'])
            story.append(chart_title)
            story.append(Spacer(1, 6))
            
            # Create image with proper sizing
            img = Image(img_buffer, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
        except Exception as e:
            error_para = Paragraph(f"<i>Chart could not be included: {str(e)}</i>", styles['Normal'])
            story.append(error_para)
            story.append(Spacer(1, 12))
    
    # Anomaly summary
    anomaly_title = Paragraph("Anomaly Summary", styles['Heading2'])
    story.append(anomaly_title)
    story.append(Spacer(1, 6))
    
    for item in report_data:
        status_icon = "●" if item['count'] > 0 else "○"
        anomaly_text = f"<b>{status_icon} {item['name']}:</b> {item['count']} anomalies detected (Threshold: {item['threshold']:.2f} {item['unit']})"
        story.append(Paragraph(anomaly_text, styles['Normal']))
    
    story.append(Spacer(1, 12))
    
    # Health report table
    if not health_report_df.empty:
        health_title = Paragraph("Health Report", styles['Heading2'])
        story.append(health_title)
        story.append(Spacer(1, 6))
        
        # Convert DataFrame to table data with proper formatting
        table_data = [health_report_df.columns.tolist()]
        for _, row in health_report_df.iterrows():
            table_data.append([str(val) for val in row.tolist()])
        
        # Calculate column widths based on content
        col_widths = [1.2*inch, 1.2*inch, 1.5*inch, 1.5*inch, 1.0*inch, 1.2*inch]
        
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Data rows styling
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid and borders
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white]),
        ]))
        story.append(table)
    
    # Footer
    story.append(Spacer(1, 24))
    footer_text = f"<i>Generated by AI-Powered Machine Diagnostics Analyzer on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    footer_para = Paragraph(footer_text, styles['Normal'])
    story.append(footer_para)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_html_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None):
    """Generate an HTML report as fallback when reportlab is not available"""
    
    # Convert chart to base64 if provided
    chart_html = ""
    if chart_fig is not None:
        try:
            img_buffer = io.BytesIO()
            chart_fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            import base64
            chart_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            chart_html = f'<img src="data:image/png;base64,{chart_b64}" style="max-width: 100%; height: auto; margin: 20px 0;">'
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
            tr:nth-child(odd) {{ background-color: white; }}
            .status-ok {{ color: #28a745; }}
            .status-warn {{ color: #ffc107; }}
            .status-error {{ color: #dc3545; }}
            .chart-container {{ text-align: center; margin: 20px 0; }}
            @media print {{ 
                body {{ margin: 20px; }} 
                .chart-container img {{ max-width: 100%; page-break-inside: avoid; }}
            }}
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
        <div class="chart-container">
            {chart_html}
        </div>
        
        <h2>Anomaly Summary</h2>
    """
    
    for item in report_data:
        status_class = "status-error" if item['count'] > 0 else "status-ok"
        status_icon = "●" if item['count'] > 0 else "○"
        html_content += f"""
        <div class="anomaly-item">
            <strong class="{status_class}">{status_icon} {item['name']}:</strong> 
            {item['count']} anomalies detected 
            (Threshold: {item['threshold']:.2f} {item['unit']})
        </div>
        """
    
    if not health_report_df.empty:
        html_content += "<h2>Health Report</h2><table>"
        # Add header
        html_content += "<tr>"
        for col in health_report_df.columns:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"
        
        # Add data rows
        for _, row in health_report_df.iterrows():
            html_content += "<tr>"
            for value in row:
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        html_content += "</table>"
    
    html_content += f"""
        <div style="margin-top: 50px; font-size: 0.9em; color: #666; text-align: center;">
            Generated by AI-Powered Machine Diagnostics Analyzer on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
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
        # This is the full list of headers found in the file
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]

        data = []
        # Data starts from row 7 (index 6)
        for r in rows[6:]:
            cells = r.findall('ss:Cell', NS)
            row_data = [cell.find('ss:Data', NS).text for cell in cells]
            data.append(row_data)
        
        if not data:
            st.error("No data found in 'Curves' worksheet.")
            return None, None
            
        # CRITICAL FIX: Determine the actual columns based on the length of the first data row.
        # This prevents a mismatch if the number of headers is different from the number of data columns.
        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]
        
        df = pd.DataFrame(data, columns=actual_columns)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df.sort_values('Crank Angle', inplace=True)

        # Return the DataFrame and the list of columns that were ACTUALLY used
        return df, actual_columns
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
                cell_text = (cells[0].find('ss:Data', NS).text or "").strip()
                if "RPM" in cell_text and "RATED" not in cell_text:
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
        levels_root = ET.fromstring(_levels_xml_content)
        # For Cylinder 1, data is in column index 2. So offset is cylinder_index + 1
        col_idx = cylinder_index + 1
        temp_str = find_xml_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', col_idx)
        
        if temp_str != "N/A":
            return f"{float(temp_str):.1f}°C"
        return "N/A"
    except (ValueError, TypeError):
        return "N/A"

@st.cache_data
def auto_discover_configuration(_source_xml_content, all_curve_names):
    """
    Automatically discovers the machine configuration from the Source.xml file
    and the provided list of available curve names. This prevents re-parsing
    and ensures the discovered curves exist in the DataFrame.
    """
    try:
        source_root = ET.fromstring(_source_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        
        machine_id = ""
        num_cylinders = 0

        # Use the robust find function to get number of cylinders
        num_cyl_str = find_xml_value(source_root, 'Source', "COMPRESSOR NUMBER OF CYLINDERS", 2)
        if num_cyl_str != "N/A":
            num_cylinders = int(num_cyl_str)
        
        machine_id_str = find_xml_value(source_root, 'Source', "Machine", 1)
        if machine_id_str != "N/A":
            machine_id = machine_id_str
        
        if num_cylinders == 0:
            st.warning("Could not determine number of cylinders from Source.xml.")
            return None

        # The list of available curves is now passed directly to the function.
        # No need to re-parse Curves.xml.
        
        cylinders_config = []
        for i in range(1, num_cylinders + 1):
            # Search for curves within the provided list of actual columns
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

# --- New, Robust Data Extraction Functions ---
def find_xml_value(root, sheet_name, partial_key, col_offset):
    """Robustly finds a value in a worksheet by row label (partial match) and column index."""
    try:
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
        if ws is None: return "N/A"
        
        rows = ws.findall('.//ss:Row', NS)
        for row in rows:
            cells = row.findall('ss:Cell', NS)
            # Check if first cell exists and contains the partial key
            if len(cells) > col_offset and cells[0].find('ss:Data', NS) is not None:
                cell_text = (cells[0].find('ss:Data', NS).text or "").strip().upper()
                # Use partial matching ('in') instead of exact ('==')
                if partial_key.upper() in cell_text:
                    value_node = cells[col_offset].find('ss:Data', NS)
                    return value_node.text if value_node is not None and value_node.text else "N/A"
        return "N/A"
    except Exception:
        return "N/A"

def generate_health_report_table(_source_xml_content, _levels_xml_content, cylinder_index):
    """Generates a DataFrame for the health report table using robust parsing."""
    try:
        source_root = ET.fromstring(_source_xml_content)
        levels_root = ET.fromstring(_levels_xml_content)
        
        # For Cylinder 1, data is in col index 2. So offset is cylinder_index + 1
        col_idx = cylinder_index + 1

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str: return "N/A"
            try:
                # Convert KPA to PSI
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str  # Return original if not a number
        
        suction_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE', col_idx))
        discharge_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE', col_idx))
        
        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'],
            'Bore (ins)': [find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx)] * 2,
            'Rod Diam (ins)': ['N/A', find_xml_value(source_root, 'Source', 'PISTON ROD DIAMETER', col_idx)],
            'Pressure Ps/Pd (psig)': [f"{suction_p} / {discharge_p}"] * 2,
            'Temp Ts/Td (°C)': [
                f"{find_xml_value(levels_root, 'Levels', 'SUCTION TEMPERATURE', col_idx)} / {find_xml_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', col_idx)}"
            ] * 2,
            'Comp. Ratio': [find_xml_value(levels_root, 'Levels', 'COMPRESSION RATIO', col_idx)] * 2,
            'Indicated Power (ihp)': [
                find_xml_value(levels_root, 'Levels', 'HEAD END INDICATED HORSEPOWER', col_idx),
                find_xml_value(levels_root, 'Levels', 'CRANK END INDICATED HORSEPOWER', col_idx)
            ]
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
                # Convert KPA to PSI
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str

        for i in range(1, num_cylinders + 1):
            col_idx = i + 1 # Column for Cyl i is at index i+1
            
            suction_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE', col_idx))
            discharge_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE', col_idx))

            detail = {
                "name": f"Cylinder {i}",
                "bore": f"{find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx)} in",
                "suction_temp": f"{find_xml_value(levels_root, 'Levels', 'SUCTION TEMPERATURE', col_idx)} °C",
                "discharge_temp": f"{find_xml_value(levels_root, 'Levels', 'DISCHARGE TEMPERATURE', col_idx)} °C",
                "suction_pressure": f"{suction_p} psig",
                "discharge_pressure": f"{discharge_p} psig",
                "flow_balance_ce": f"{find_xml_value(levels_root, 'Levels', 'CRANK END FLOW BALANCE', col_idx)} %",
                "flow_balance_he": f"{find_xml_value(levels_root, 'Levels', 'HEAD END FLOW BALANCE', col_idx)} %"
            }

            # Clean up display for N/A values
            for key, value in detail.items():
                if "N/A" in str(value):
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
            # Show progress indicator
            with st.spinner("Analyzing machine data..."):
                # Step 1: Load data and get the *actual* columns used in the DataFrame.
                df, actual_curve_names = load_all_curves_data(files_content['curves'])
                
                # Step 2: Pass the actual columns to the discovery function to ensure sync.
                if df is not None and actual_curve_names is not None:
                    discovered_config = auto_discover_configuration(files_content['source'], actual_curve_names)
                else:
                    # If df loading fails, discovered_config remains None
                    discovered_config = None
            
            if df is not None and discovered_config:
                rpm = extract_rpm(files_content['levels'])
                machine_id = discovered_config.get('machine_id', 'N/A')
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
                        with st.expander("Add labels to detected anomalies and valve events", expanded=True):
                            st.subheader("Fault Labels")
                            for item in report_data:
                                if item['count'] > 0:
                                    analysis_id = analysis_ids[item['name']]
                                    
                                    with st.form(key=f"label_form_{analysis_id}"):
                                        st.write(f"**{item['name']} Anomaly** ({item['count']} points detected)")
                                        user_label = st.text_input(
                                            "Enter fault label:",
                                            key=f"txt_label_{analysis_id}",
                                            placeholder="e.g., Valve sticking, Pressure spike, etc."
                                        )
                                        
                                        submitted = st.form_submit_button("Save Label")
                                        
                                        if submitted:
                                            if user_label.strip():
                                                try:
                                                    cursor.execute(
                                                        "INSERT INTO labels (analysis_id, label_text) VALUES (?, ?)",
                                                        (analysis_id, user_label.strip())
                                                    )
                                                    db_conn.commit()
                                                    st.success(f"✅ Label saved: '{user_label}'")
                                                except Exception as e:
                                                    st.error(f"❌ Error saving label: {str(e)}")
                                            else:
                                                st.warning("⚠️ Please enter a label before saving.")
                                    
                                    # Show existing labels for this analysis
                                    existing_labels = cursor.execute(
                                        "SELECT label_text FROM labels WHERE analysis_id = ?",
                                        (analysis_id,)
                                    ).fetchall()
                                    
                                    if existing_labels:
                                        labels_text = ", ".join([label[0] for label in existing_labels])
                                        st.info(f"📝 Existing labels: {labels_text}")
                            
                            st.subheader("Mark Valve Open/Close Events")
                            for item in report_data:
                                if item['name'] != 'Pressure':  # Only for valves
                                    analysis_id = analysis_ids[item['name']]
                                    
                                    # Use a form to prevent automatic reruns
                                    with st.form(key=f"valve_form_{analysis_id}"):
                                        st.write(f"**{item['name']} Valve Events:**")
                                        
                                        cols = st.columns([1, 1])
                                        with cols[0]:
                                            open_angle = st.number_input(
                                                "Open Angle (degrees)", 
                                                key=f"open_{analysis_id}",
                                                value=None,
                                                format="%.2f",
                                                help="Enter the crank angle where the valve opens"
                                            )
                                        with cols[1]:
                                            close_angle = st.number_input(
                                                "Close Angle (degrees)", 
                                                key=f"close_{analysis_id}",
                                                value=None,
                                                format="%.2f",
                                                help="Enter the crank angle where the valve closes"
                                            )
                                        
                                        # Form submit button - only triggers on explicit submission
                                        submitted = st.form_submit_button(f"Save Events for {item['name']}")
                                        
                                        if submitted:
                                            # Validate angles
                                            valid_open = validate_angle(open_angle, f"{item['name']} Open")
                                            valid_close = validate_angle(close_angle, f"{item['name']} Close")
                                            
                                            if valid_open and valid_close:
                                                try:
                                                    # Clear existing events for this analysis
                                                    cursor.execute("DELETE FROM valve_events WHERE analysis_id = ?", (analysis_id,))
                                                    
                                                    events_saved = []
                                                    if open_angle is not None:
                                                        cursor.execute(
                                                            "INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)",
                                                            (analysis_id, 'open', open_angle)
                                                        )
                                                        events_saved.append(f"Open at {open_angle}°")
                                                    
                                                    if close_angle is not None:
                                                        cursor.execute(
                                                            "INSERT INTO valve_events (analysis_id, event_type, crank_angle) VALUES (?, ?, ?)",
                                                            (analysis_id, 'close', close_angle)
                                                        )
                                                        events_saved.append(f"Close at {close_angle}°")
                                                    
                                                    db_conn.commit()
                                                    
                                                    if events_saved:
                                                        st.success(f"✅ Events saved for {item['name']}: {', '.join(events_saved)}")
                                                    else:
                                                        st.warning("ℹ️ No events to save. Please enter angle values.")
                                                        
                                                except Exception as e:
                                                    st.error(f"❌ Error saving events: {str(e)}")
                                    
                                    # Show current events for this valve (outside the form)
                                    current_events = cursor.execute(
                                        "SELECT event_type, crank_angle FROM valve_events WHERE analysis_id = ? ORDER BY crank_angle",
                                        (analysis_id,)
                                    ).fetchall()
                                    
                                    if current_events:
                                        events_text = ", ".join([f"{event[0].capitalize()}: {event[1]}°" for event in current_events])
                                        st.info(f"🔧 Current events for {item['name']}: {events_text}")
                                    
                                    st.divider()  # Visual separator between valves

                        st.header("📝 Diagnostic Summary")
                        cylinder_index = int(re.search(r'\d+', selected_cylinder_name).group())
                        discharge_temp = extract_temperature(files_content['levels'], cylinder_index)
                        st.markdown(f"**Machine ID:** {machine_id} | **Operating RPM:** {rpm} | **Discharge Temp:** {discharge_temp}")
                        st.markdown(f"**Data Points Analyzed:** {len(df):,}")
                        st.markdown("--- \n ### Anomaly Summary")
                        for item in report_data:
                            status_icon = "🔴" if item['count'] > 0 else "🟢"
                            st.markdown(f"{status_icon} **{item['name']} Anomalies:** {item['count']} points (Threshold: {item['threshold']:.2f} {item['unit']})")

                        st.header("📋 Compressor Health Report")
                        health_report_df = generate_health_report_table(files_content['source'], files_content['levels'], cylinder_index)
                        if not health_report_df.empty:
                            st.dataframe(health_report_df, use_container_width=True, hide_index=True)
                        
                        # PDF Generation Section
                        st.header("📄 Export Report")
                        
                        if REPORTLAB_AVAILABLE:
                            button_text = "🔄 Generate PDF Report"
                            file_extension = "pdf"
                            mime_type = "application/pdf"
                        else:
                            button_text = "🔄 Generate HTML Report"
                            file_extension = "html"
                            mime_type = "text/html"
                            st.info("💡 Install `reportlab` for PDF generation: `pip install reportlab`")
                        
                        if st.button(button_text, type="primary"):
                            try:
                                with st.spinner("Generating report..."):
                                    report_buffer = generate_pdf_report(machine_id, rpm, selected_cylinder_name, report_data, health_report_df, fig)
                                
                                download_label = f"📥 Download {file_extension.upper()} Report"
                                filename = f"diagnostics_report_{machine_id}_{selected_cylinder_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
                                
                                st.download_button(
                                    label=download_label,
                                    data=report_buffer.getvalue(),
                                    file_name=filename,
                                    mime=mime_type
                                )
                                st.success(f"✅ {file_extension.upper()} report generated successfully!")
                            except Exception as e:
                                st.error(f"❌ Error generating report: {str(e)}")
                        
                        st.header("🔧 Cylinder Details")
                        all_details = get_all_cylinder_details(files_content['source'], files_content['levels'], len(cylinders))
                        if all_details:
                            cols = st.columns(len(all_details))
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
                # Error messages are already shown in the loading functions
                pass
            else:
                st.error("Could not discover a valid machine configuration from the provided files.")

        except Exception as e:
            st.error(f"❌ An error occurred during processing. Please check the files. Details: {e}")
            st.exception(e)  # Show full traceback for debugging
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
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = labels_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Labels as CSV",
                data=csv_data,
                file_name=f"anomaly_labels_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = labels_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download Labels as JSON",
                data=json_data,
                file_name=f"anomaly_labels_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Summary statistics
        st.subheader("📊 Labels Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Labels", len(all_labels))
        with col2:
            unique_machines = labels_df['Machine ID'].nunique()
            st.metric("Unique Machines", unique_machines)
        with col3:
            unique_cylinders = labels_df['Cylinder'].nunique()
            st.metric("Unique Cylinders", unique_cylinders)
    else:
        st.info("📝 No labels found. Start analyzing data and adding labels to build your diagnostic knowledge base.")

except Exception as e:
    st.error(f"❌ Error retrieving labels: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    🔧 AI-Powered Machine Diagnostics Analyzer | 
    Session ID: {} | 
    Last Updated: {}
</div>
""".format(
    st.session_state.active_session_id or "None", 
    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
), unsafe_allow_html=True)
