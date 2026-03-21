"""
Database query helpers.

Streamlit is imported for st.error / st.warning display inside save functions.
The db_client is always passed as a parameter — never imported globally.
"""

import datetime
import pandas as pd
import streamlit as st
import plotly.express as px


def get_last_row_id(client):
    rs = client.execute("SELECT last_insert_rowid()")
    return rs.rows[0][0] if rs.rows else None


def save_valve_event_to_db(db_client, session_id, cylinder_name, curve_name, crank_angle, event_type="Manual Tag", fault_classification=None):
    """Save valve event to database with fault classification."""
    try:
        db_client.execute(
            "INSERT INTO valve_events (session_id, cylinder_name, curve_name, crank_angle, data_value, curve_type) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, cylinder_name, curve_name, crank_angle, 0.0, event_type)
        )
    except Exception as e:
        st.error(f"Database error saving valve event: {e}")
        raise


def save_anomaly_tag_to_db(db_client, session_id, cylinder_name, curve_name, crank_angle, fault_classification, tag_type="Manual Tag"):
    """Save anomaly tag to database - separate from valve events."""
    try:
        db_client.execute(
            "INSERT INTO anomaly_tags (session_id, cylinder_name, curve_name, crank_angle, fault_classification, tag_type) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, cylinder_name, curve_name, crank_angle, fault_classification, tag_type)
        )
    except Exception as e:
        st.error(f"Database error saving anomaly tag: {e}")
        raise


def save_waveform_data_to_db(db_client, session_id, cylinder_name, curve_name, crank_angles, data_values, curve_type):
    """
    Save waveform data points to database for ML training.
    Uses batch insert for efficiency with parameterized statements.

    Args:
        db_client: Database client connection
        session_id: Current session ID
        cylinder_name: Name of the cylinder
        curve_name: Name of the curve (e.g., "HE PT", "CE SV")
        crank_angles: Array of crank angle values
        data_values: Array of corresponding data values (pressure/vibration)
        curve_type: Type of curve ("pressure" or "vibration")
    """
    try:
        # First, check if waveform data already exists for this session/cylinder/curve
        check_rs = db_client.execute(
            "SELECT COUNT(*) FROM waveform_data WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
            (session_id, cylinder_name, curve_name)
        )
        existing_count = check_rs.rows[0][0] if check_rs.rows else 0

        if existing_count > 0:
            # Data already saved, skip to avoid duplicates
            return

        # Prepare batch insert statements using parameterized queries
        insert_statements = []
        for angle, value in zip(crank_angles, data_values):
            insert_statements.append(
                (
                    "INSERT INTO waveform_data (session_id, cylinder_name, curve_name, crank_angle, data_value, curve_type) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, cylinder_name, curve_name, float(angle), float(value), curve_type)
                )
            )

        # Execute in batches of 500 to avoid overwhelming the database
        batch_size = 500
        for i in range(0, len(insert_statements), batch_size):
            batch = insert_statements[i:i + batch_size]
            # libsql_client batch accepts a list of (statement, params) tuples
            db_client.batch(list(batch))

    except Exception as e:
        st.warning(f"Could not save waveform data for {curve_name}: {e}")
        # Don't raise - waveform saving is optional and shouldn't break the app


def display_historical_analysis(db_client):
    """
    Queries the database for historical data and displays it as a trend chart.
    """
    st.subheader("Anomaly Count Trend Over Time")

    query = """
        SELECT
            s.timestamp,
            s.machine_id,
            SUM(a.anomaly_count) as total_anomalies
        FROM analyses a
        JOIN sessions s ON a.session_id = s.id
        GROUP BY s.id, s.timestamp, s.machine_id
        ORDER BY s.timestamp ASC
    """
    try:
        rs = db_client.execute(query)
        if not rs.rows:
            st.info("No historical analysis data found to display.")
            return

        # Create DataFrame with manual column names
        df = pd.DataFrame(rs.rows, columns=['timestamp', 'machine_id', 'total_anomalies'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if df.empty:
            st.info("No historical analysis data found to display.")
            return

        # Create the plot
        fig = px.line(
            df,
            x='timestamp',
            y='total_anomalies',
            color='machine_id',
            markers=True,
            title='Total Anomaly Count by Machine Over Time',
            labels={
                "timestamp": "Date of Analysis",
                "total_anomalies": "Total Anomalies Found",
                "machine_id": "Machine ID"
            }
        )
        fig.update_layout(template="ggplot2")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load historical data: {e}")


def upsert_analysis_records(db_client, session_id, cylinder_name, report_data):
    """
    Insert or update one analyses row per item in report_data.
    Returns a dict mapping item['name'] -> analysis_id.
    """
    analysis_ids = {}
    for item in report_data:
        rs = db_client.execute(
            "SELECT id FROM analyses WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
            (session_id, cylinder_name, item['curve_name'])
        )
        existing_row = rs.rows[0] if rs.rows else None
        if existing_row:
            analysis_id = existing_row[0]
            db_client.execute(
                "UPDATE analyses SET anomaly_count = ?, threshold = ? WHERE id = ?",
                (item['count'], item['threshold'], analysis_id)
            )
        else:
            db_client.execute(
                "INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)",
                (session_id, cylinder_name, item['curve_name'], item['count'], item['threshold'])
            )
            analysis_id = get_last_row_id(db_client)
        analysis_ids[item['name']] = analysis_id
    return analysis_ids


def check_and_display_alerts(db_client, machine_id, cylinder_name, critical_alerts, health_score):
    """Check for critical conditions and display in-app alerts."""
    current_time = datetime.datetime.now()

    # Health score alert
    if health_score < 40:
        alert_msg = f"Health score critically low: {health_score:.1f}"
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'CRITICAL', alert_msg, current_time)
            )
        except Exception:
            pass
        st.error(f"CRITICAL ALERT: {alert_msg}")

    elif health_score < 60:
        alert_msg = f"Health score below normal: {health_score:.1f}"
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'WARNING', alert_msg, current_time)
            )
        except Exception:
            pass
        st.warning(f"WARNING: {alert_msg}")

    # Critical anomaly alerts
    for alert in critical_alerts:
        try:
            db_client.execute(
                "INSERT INTO alerts (machine_id, cylinder, severity, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (machine_id, cylinder_name, 'HIGH', alert, current_time)
            )
        except Exception:
            pass
        st.error(f"{alert}")
