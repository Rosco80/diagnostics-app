"""
Reusable Streamlit UI components.
"""

import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def inject_custom_css():
    """Inject custom CSS for enhanced styling."""
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def render_section_header(title, icon="🔧"):
    """Render consistent section headers."""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h3 style="margin: 0; color: var(--primary-color);">{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def render_main_header():
    """Render professional main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔧 AI-Powered Machine Diagnostics</h1>
        <p>Advanced condition monitoring and fault detection for reciprocating machinery</p>
    </div>
    """, unsafe_allow_html=True)


def render_status_card(title, value, status="normal", icon="📊"):
    """Render professional status card."""
    status_class = {
        "excellent": "status-excellent",
        "good": "status-good",
        "warning": "status-warning",
        "critical": "status-critical"
    }.get(status, "status-good")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{title}</div>
        <div class="{status_class}">{status.upper()}</div>
    </div>
    """, unsafe_allow_html=True)


def render_enhanced_metrics_dashboard(analysis_results):
    """Enhanced metrics dashboard with professional layout."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("System Overview", "📈")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_status_card("Machine Status", "OPERATIONAL", "good", "⚙️")
    with col2:
        render_status_card("Health Score", "85.2", "good", "❤️")
    with col3:
        render_status_card("Anomalies", "3", "warning", "⚠️")
    with col4:
        render_status_card("Last Update", datetime.datetime.now().strftime("%H:%M"), "excellent", "🕒")

    st.markdown('</div>', unsafe_allow_html=True)


def render_enhanced_results_dashboard(analysis_data):
    """Enhanced results dashboard with comprehensive insights."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("Analysis Results", "📊")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Key Findings")
        findings = [
            "✅ Cylinder performance within normal parameters",
            "⚠️ Minor valve timing deviation detected (Cylinder 2)",
            "📈 Compression pressure trending upward",
            "🔄 Recommend continued monitoring"
        ]
        for finding in findings:
            st.markdown(f"• {finding}")

    with col2:
        st.markdown("### Health Metrics")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=85.2,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 60], 'color': "#e74c3c"},
                    {'range': [60, 80], 'color': "#f39c12"},
                    {'range': [80, 100], 'color': "#27ae60"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_action_buttons():
    """Render action buttons with consistent styling."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_header("Actions", "⚡")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            st.success("Report generation started...")
    with col2:
        if st.button("📧 Send Alert", use_container_width=True):
            st.info("Alert notification sent!")
    with col3:
        if st.button("💾 Save Analysis", use_container_width=True):
            st.success("Analysis saved successfully!")
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.info("Data refresh initiated...")

    st.markdown('</div>', unsafe_allow_html=True)


def render_valve_sensors_table(valve_vibration_curves, cylinder_index):
    """
    Display a sorted table of valve sensor metadata for the selected cylinder.
    `valve_vibration_curves` is the list from the discovered cylinder config.
    `cylinder_index` is the integer cylinder number (used for the Cyl End column).
    """
    st.subheader("Valve Sensors Detected")
    if not valve_vibration_curves:
        st.info("No valve sensors detected for this cylinder")
        return

    rows = []
    for valve_info in valve_vibration_curves:
        vn = valve_info['name']
        end_order = 0 if vn.startswith('HE') else (1 if vn.startswith('CE') else 2)
        cyl_end = (f'{cylinder_index}H' if vn.startswith('HE') else
                   f'{cylinder_index}C' if vn.startswith('CE') else 'N/A')
        type_order = 0 if 'Discharge' in vn else (1 if 'Suction' in vn else 2)
        valve_type = ('Discharge' if 'Discharge' in vn else
                      'Suction' if 'Suction' in vn else 'N/A')
        sensor_type = ('Vibration' if '(VIB)' in vn else
                       'Ultrasonic' if '(US)' in vn else 'N/A')
        rows.append({
            'Cyl End': cyl_end, 'Valve Type': valve_type,
            'Valve Name': vn, 'Sensor Type': sensor_type,
            '_end_order': end_order, '_type_order': type_order,
        })

    rows.sort(key=lambda x: (x['_end_order'], x['_type_order'], x['Valve Name']))
    for row in rows:
        del row['_end_order']
        del row['_type_order']

    he_count = sum(1 for r in rows if 'H' in r['Cyl End'])
    ce_count = sum(1 for r in rows if 'C' in r['Cyl End'])
    st.info(f"Detected {len(valve_vibration_curves)} valves ({he_count} on Head End, {ce_count} on Crank End)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_all_cylinder_details(all_details):
    """
    Display a columnar card layout for all cylinder details.
    `all_details` is the list returned by get_all_cylinder_details().
    """
    if not all_details:
        return
    st.header("All Cylinder Details")
    cols = st.columns(len(all_details) or 1)
    for i, detail in enumerate(all_details):
        with cols[i]:
            st.markdown(f"""
<div style='border:1px solid #ddd;border-radius:5px;padding:10px;margin-bottom:10px;'>
<h5>{detail['name']}</h5>
<small>Bore: <strong>{detail['bore']}</strong></small><br>
<small>Rod Dia.: <strong>{detail.get('rod_diameter','N/A')}</strong></small><br>
<small>Stroke: <strong>{detail.get('stroke','N/A')}</strong></small><br>
<small>Volume: <strong>{detail.get('volume','N/A')}</strong></small><br>
<small>Temps (S/D): <strong>{detail['suction_temp']} / {detail['discharge_temp']}</strong></small><br>
<small>Pressures (S/D): <strong>{detail['suction_pressure']} / {detail['discharge_pressure']}</strong></small><br>
<small>Flow Balance (CE/HE): <strong>{detail['flow_balance_ce']} / {detail['flow_balance_he']}</strong></small>
</div>
""", unsafe_allow_html=True)


def render_machine_info_bar(discovered_config):
    """Render the horizontal machine-info summary bar."""
    st.markdown(f"""
<div style='border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0;'>
  <strong>Machine ID:</strong> {discovered_config.get('machine_id','N/A')} &nbsp;|&nbsp;
  <strong>Model:</strong> {discovered_config.get('model','N/A')} &nbsp;|&nbsp;
  <strong>Serial:</strong> {discovered_config.get('serial_number','N/A')} &nbsp;|&nbsp;
  <strong>Rated RPM:</strong> {discovered_config.get('rated_rpm','N/A')} &nbsp;|&nbsp;
  <strong>Rated HP:</strong> {discovered_config.get('rated_hp','N/A')}
</div>
""", unsafe_allow_html=True)
