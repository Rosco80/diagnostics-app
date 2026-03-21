"""
Analytics and data-export UI sections:
  - Historical trend chart
  - Saved labels viewer
  - Phase 2 training data export
  - Data integrity / readiness check
"""

import streamlit as st
import pandas as pd

from database.queries import display_historical_analysis

# PRD Section 7.1 minimum sample requirements per fault class
PRD_REQUIREMENTS = {
    "Valve Opening": 100,
    "Valve Leakage": 50,
    "Valve Sticking": 25,
    "Closing Hard or Slamming": 25,
    "Broken or Missing Valve Parts": 20,
    "Spring Fatigue or Failure": 15,
    "Other": 10,
}


def render_labels_filter_sidebar(db_client):
    """
    Render the machine-ID filter selectbox in the sidebar.
    Returns the selected machine ID string (may be "All").
    """
    st.header("3. View All Saved Labels")
    rs = db_client.execute("SELECT DISTINCT machine_id FROM sessions ORDER BY machine_id ASC")
    machine_id_options = [row[0] for row in rs.rows]
    selected = st.selectbox("Filter labels by Machine ID", options=["All"] + machine_id_options)
    return selected


def render_historical_trend(db_client):
    """Historical anomaly trend chart."""
    st.markdown("---")
    st.header("Historical Trend Analysis")
    display_historical_analysis(db_client)


def render_saved_labels(db_client, machine_id_filter):
    """Saved-labels table with CSV download."""
    st.header("All Saved Labels")
    query = (
        "SELECT s.timestamp, s.machine_id, a.cylinder_name, a.curve_name, l.label_text "
        "FROM labels l JOIN analyses a ON l.analysis_id = a.id JOIN sessions s ON a.session_id = s.id"
    )
    params = []
    if machine_id_filter != "All":
        query += " WHERE s.machine_id = ?"
        params.append(machine_id_filter)
    query += " ORDER BY s.timestamp DESC"

    rs = db_client.execute(query, tuple(params))
    if rs.rows:
        labels_df = pd.DataFrame(rs.rows, columns=['Timestamp', 'Machine ID', 'Cylinder', 'Curve', 'Label'])
        st.dataframe(labels_df, use_container_width=True)
        csv_data = labels_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Labels as CSV", csv_data, "anomaly_labels.csv", "text/csv")
    else:
        st.info("No labels found.")


def render_training_data_export(db_client, machine_id_filter):
    """Phase 2 training data export: fault tags + combined waveform dataset."""
    st.markdown("---")
    st.header("Phase 2: AI Training Data Export")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Manual Fault Tags")
        tags_query = """
            SELECT
                s.timestamp,
                s.machine_id,
                at.cylinder_name,
                at.curve_name,
                at.crank_angle,
                at.fault_classification,
                at.tag_type,
                at.created_at
            FROM anomaly_tags at
            JOIN sessions s ON at.session_id = s.id
        """
        tags_params = []
        if machine_id_filter != "All":
            tags_query += " WHERE s.machine_id = ?"
            tags_params.append(machine_id_filter)
        tags_query += " ORDER BY at.created_at DESC"

        tags_rs = db_client.execute(tags_query, tuple(tags_params))
        if tags_rs.rows:
            tags_df = pd.DataFrame(tags_rs.rows, columns=[
                'Session Timestamp', 'Machine ID', 'Cylinder', 'Curve',
                'Crank Angle', 'Fault Classification', 'Tag Type', 'Created At'
            ])
            st.dataframe(tags_df, use_container_width=True)
            tags_csv = tags_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Fault Tags CSV",
                tags_csv,
                "fault_tags_training_data.csv",
                "text/csv",
                key="download_tags"
            )
            st.markdown("**Tag Statistics:**")
            for fault_type, count in tags_df['Fault Classification'].value_counts().items():
                st.write(f"- {fault_type}: {count} tags")
        else:
            st.info("No manual tags found. Use the tagging interface above to classify anomalies.")

    with col2:
        st.subheader("Combined Training Dataset")
        st.markdown("**Waveform data + Fault classifications**")

        combined_query = """
            SELECT
                s.machine_id,
                s.timestamp,
                wd.cylinder_name,
                wd.curve_name,
                wd.crank_angle,
                wd.data_value,
                wd.curve_type,
                at.fault_classification
            FROM waveform_data wd
            JOIN sessions s ON wd.session_id = s.id
            JOIN anomaly_tags at ON (
                wd.session_id = at.session_id AND
                wd.cylinder_name = at.cylinder_name AND
                wd.curve_name = at.curve_name AND
                ABS(wd.crank_angle - at.crank_angle) < 1.0
            )
        """
        combined_params = []
        if machine_id_filter != "All":
            combined_query += " WHERE s.machine_id = ?"
            combined_params.append(machine_id_filter)
        combined_query += " ORDER BY s.timestamp DESC, wd.crank_angle ASC LIMIT 10000"

        combined_rs = db_client.execute(combined_query, tuple(combined_params))
        if combined_rs.rows:
            combined_df = pd.DataFrame(combined_rs.rows, columns=[
                'Machine ID', 'Session Time', 'Cylinder', 'Curve',
                'Crank Angle', 'Value', 'Type', 'Fault Classification'
            ])
            st.write(f"**{len(combined_df):,} data points** with fault labels ready for training")
            st.dataframe(combined_df.head(100), use_container_width=True)
            combined_csv = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Training Dataset CSV",
                combined_csv,
                "ml_training_dataset.csv",
                "text/csv",
                key="download_combined"
            )
        else:
            st.info("No combined training data available yet. Save tags with waveforms to generate training data.")


def render_data_integrity_check(db_client):
    """Data integrity metrics and PRD training-readiness progress bars."""
    st.markdown("---")
    st.header("Data Integrity & Readiness Check")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rs = db_client.execute("SELECT COUNT(*) FROM sessions")
        st.metric("Total Sessions", rs.rows[0][0] if rs.rows else 0)
    with c2:
        rs = db_client.execute("SELECT COUNT(*) FROM anomaly_tags")
        st.metric("Manual Tags", rs.rows[0][0] if rs.rows else 0)
    with c3:
        rs = db_client.execute("SELECT COUNT(DISTINCT session_id || cylinder_name || curve_name) FROM waveform_data")
        st.metric("Waveform Datasets", rs.rows[0][0] if rs.rows else 0)
    with c4:
        rs = db_client.execute("SELECT COUNT(*) FROM analyses")
        st.metric("Analyses", rs.rows[0][0] if rs.rows else 0)

    st.subheader("Phase 2 Training Data Requirements (PRD Section 7.1)")

    fault_dist_rs = db_client.execute(
        "SELECT fault_classification, COUNT(*) FROM anomaly_tags GROUP BY fault_classification"
    )
    fault_distribution = {row[0]: row[1] for row in fault_dist_rs.rows} if fault_dist_rs.rows else {}

    for fault_type, required in PRD_REQUIREMENTS.items():
        current = fault_distribution.get(fault_type, 0)
        percentage = min(100, int((current / required) * 100))
        marker = "G" if current >= required else ("Y" if current >= required * 0.5 else "R")
        st.write(f"[{marker}] **{fault_type}**: {current}/{required} ({percentage}%)")
        st.progress(percentage / 100)

    total_tags = sum(fault_distribution.values())
    if total_tags >= 200:
        st.success("Sufficient data collected for Phase 2 supervised learning!")
    else:
        st.warning(f"Need {200 - total_tags} more tagged examples to meet minimum Phase 2 requirements.")
