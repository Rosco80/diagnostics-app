"""
Sidebar rendering: cylinder selection, pressure options, AI model tuning.
Also contains the valve-timing config expander and PDF report generation
button, which are rendered in the main content area alongside the cylinder
view but logically belong to cylinder-level controls.
"""

import streamlit as st

from database.queries import get_last_row_id


def render_cylinder_selection_sidebar(cylinders_config):
    """
    Fixed cylinder selection that always defaults to Cylinder 1.
    """
    cylinders = cylinders_config.get("cylinders", [])
    cylinder_names = [c.get("cylinder_name") for c in cylinders]

    if not cylinder_names:
        st.sidebar.error("No cylinders detected")
        return None, None

    # Find default index for Cylinder 1
    default_index = 0
    if "Cylinder 1" in cylinder_names:
        default_index = cylinder_names.index("Cylinder 1")

    selected_cylinder_name = st.sidebar.selectbox(
        "Select Cylinder for Detailed View",
        cylinder_names,
        index=default_index,  # This ensures proper default selection
        help="Choose which cylinder to analyze in detail"
    )

    selected_cylinder_config = next(
        (c for c in cylinders if c.get("cylinder_name") == selected_cylinder_name),
        None
    )
    st.session_state['selected_cylinder_name'] = selected_cylinder_name

    # Valve end filter — lets user show only HE, CE, or all valves
    if selected_cylinder_config:
        valve_curves = selected_cylinder_config.get('valve_vibration_curves', [])
        # Detect if both HE and CE valves exist (look for ' H ' and ' C ' in curve names)
        has_he = any(' H ' in v.get('curve', '') for v in valve_curves)
        has_ce = any(' C ' in v.get('curve', '') for v in valve_curves)
        if has_he and has_ce:
            valve_end_filter = st.sidebar.radio(
                "Valve End Filter",
                ["All", "HE only", "CE only"],
                index=0,
                key='valve_end_filter',
                horizontal=True,
            )
            if valve_end_filter != "All":
                marker = ' H ' if valve_end_filter == "HE only" else ' C '
                filtered = [v for v in valve_curves if marker in v.get('curve', '')]
                selected_cylinder_config = dict(selected_cylinder_config)
                selected_cylinder_config['valve_vibration_curves'] = filtered

    return selected_cylinder_name, selected_cylinder_config


def render_pressure_options_sidebar():
    """
    Render pressure analysis options similar to professional software with signal validation.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Pressure Analysis Options")

    # Main pressure enable toggle
    enable_pressure = st.sidebar.checkbox(
        "Enable Advanced Pressure Analysis",
        value=True,
        key='enable_pressure'
    )

    pressure_options = {'enable_pressure': enable_pressure}

    if enable_pressure:
        # Pressure Traces Section
        st.sidebar.markdown("**Pressure Traces:**")
        pressure_options['show_he_pt'] = st.sidebar.checkbox("Show HE PT trace", key='show_he_pt')
        pressure_options['show_ce_pt'] = st.sidebar.checkbox("Show CE PT trace", value=True, key='show_ce_pt')

        # Period Selection
        st.sidebar.markdown("**Pressure Period Selection:**")
        period_options = [
            "Median", "Average", "Maximum", "Minimum",
            "Outer Envelope", "Inner Envelope", "All periods"
        ]

        pressure_options['period_selection'] = st.sidebar.selectbox(
            "Period Selection:",
            period_options,
            index=0,  # Default to Median
            key='pressure_period'
        )

        # Additional options
        pressure_options['use_crc_data'] = st.sidebar.checkbox("Use CRC data", key='use_crc_data')
    else:
        # Set all options to False if pressure analysis is disabled
        pressure_options.update({
            'show_he_pt': False,
            'show_ce_pt': False,
            'period_selection': "Median",
            'use_crc_data': False
        })

    return pressure_options


def render_ai_model_tuning_section(db_client, discovered_config):
    """Enhanced AI Model Tuning with machine-specific configuration."""
    st.markdown("---")
    st.subheader("AI Model Tuning")

    # Get machine ID if available
    machine_id = discovered_config.get('machine_id', 'N/A') if discovered_config else None

    if machine_id and machine_id != 'N/A':
        st.markdown(f"**Machine-Specific Configuration** for `{machine_id}`")

        # Load existing config from database
        try:
            rs = db_client.execute("SELECT * FROM configs WHERE machine_id = ?", (machine_id,))
            existing_config = rs.rows[0] if rs.rows else None
        except Exception as e:
            st.warning(f"Could not load saved configuration: {str(e)}")
            existing_config = None

        # Configuration inputs with saved values
        col1, col2 = st.columns(2)

        with col1:
            contamination_level = st.slider(
                "Anomaly Detection Sensitivity",
                min_value=0.01,
                max_value=0.20,
                value=existing_config[1] if existing_config else 0.05,
                step=0.01,
                help="Machine-specific sensitivity for anomaly detection"
            )

            pressure_limit = st.number_input(
                "Pressure Anomaly Threshold",
                min_value=1,
                max_value=50,
                value=existing_config[2] if existing_config else 10,
                help="Maximum allowed pressure anomalies before alert"
            )

        with col2:
            valve_limit = st.number_input(
                "Valve Anomaly Threshold",
                min_value=1,
                max_value=30,
                value=existing_config[3] if existing_config else 5,
                help="Maximum allowed valve anomalies before alert"
            )

            # Save button
            if st.button("💾 Save Configuration", type="primary"):
                try:
                    db_client.execute(
                        "INSERT OR REPLACE INTO configs (machine_id, contamination, pressure_anom_limit, valve_anom_limit, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                        (machine_id, contamination_level, pressure_limit, valve_limit)
                    )
                    st.success(f"✅ Configuration saved for {machine_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")

        # Show if config was loaded
        if existing_config:
            st.info(f"📋 Using saved configuration (last updated: {existing_config[4] if len(existing_config) > 4 else 'Unknown'})")
        else:
            st.info("🆕 Using default values - save to create machine-specific configuration")

        return contamination_level, pressure_limit, valve_limit

    else:
        # Fallback to original slider when no machine ID
        st.info("ℹ️ Upload files to enable machine-specific configuration")
        contamination_level = st.slider(
            "Anomaly Detection Sensitivity",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Adjust the proportion of data points considered as anomalies. Higher values mean more sensitive detection."
        )
        return contamination_level, 10, 5  # Default limits


def render_valve_timing_config(db_client, report_data, selected_cylinder_name):
    """
    Expander for configuring valve open/close crank-angle events.
    Loads existing events from the database and saves new ones on submit.
    """
    with st.expander("Configure Valve Open/Close Events"):
        st.info("Set valve timing for each curve. Values are in crank angle degrees.")

        valve_data = {}
        for item in report_data:
            if item['name'] != 'Pressure':
                events_rs = db_client.execute(
                    "SELECT curve_type, crank_angle FROM valve_events WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
                    (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'])
                )
                events = {e[0]: e[1] for e in events_rs.rows}
                valve_data[item['name']] = {
                    'curve_name': item['curve_name'],
                    'open': events.get('open'),
                    'close': events.get('close')
                }

        with st.form("all_valve_events"):
            st.markdown("**Valve Timing Configuration**")
            for valve_name, data in valve_data.items():
                cols = st.columns([3, 2, 2])
                cols[0].write(f"**{valve_name}**")
                open_val = cols[1].number_input(
                    "Open deg",
                    value=data['open'],
                    key=f"open_{valve_name}",
                    format="%.2f",
                    help="Valve opening angle"
                )
                close_val = cols[2].number_input(
                    "Close deg",
                    value=data['close'],
                    key=f"close_{valve_name}",
                    format="%.2f",
                    help="Valve closing angle"
                )
                valve_data[valve_name]['open_input'] = open_val
                valve_data[valve_name]['close_input'] = close_val

            if st.form_submit_button("Save All Valve Events", type="primary"):
                try:
                    saved_count = 0
                    for valve_name, data in valve_data.items():
                        db_client.execute(
                            "DELETE FROM valve_events WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
                            (st.session_state.active_session_id, selected_cylinder_name, data['curve_name'])
                        )
                        if data['open_input'] is not None:
                            db_client.execute(
                                "INSERT INTO valve_events (session_id, cylinder_name, curve_name, crank_angle, data_value, curve_type) VALUES (?, ?, ?, ?, ?, ?)",
                                (st.session_state.active_session_id, selected_cylinder_name, data['curve_name'], data['open_input'], 0.0, 'open')
                            )
                            saved_count += 1
                        if data['close_input'] is not None:
                            db_client.execute(
                                "INSERT INTO valve_events (session_id, cylinder_name, curve_name, crank_angle, data_value, curve_type) VALUES (?, ?, ?, ?, ?, ?)",
                                (st.session_state.active_session_id, selected_cylinder_name, data['curve_name'], data['close_input'], 0.0, 'close')
                            )
                            saved_count += 1
                    st.success(f"Saved {saved_count} valve events successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save valve events: {str(e)}")


def render_pdf_export(
    db_client, machine_id, rpm, selected_cylinder_name,
    report_data, health_report_df, fig, suggestions,
    health_score, critical_alerts,
    generate_pdf_fn,
):
    """
    'Generate Report' button + download button for the PDF export.
    `generate_pdf_fn` is the callable imported from reporting.pdf.
    """
    if st.button("Generate Report for this Cylinder", type="primary", key='gen_report'):
        plot_key = f"{selected_cylinder_name.replace(' ', '_')}_plot"
        current_tagged_events = st.session_state.valve_event_tags.get(plot_key, [])
        pdf_buffer = generate_pdf_fn(
            machine_id, rpm, selected_cylinder_name, report_data,
            health_report_df, fig, suggestions, health_score,
            critical_alerts, current_tagged_events
        )
        if pdf_buffer:
            st.download_button(
                "Download PDF Report",
                pdf_buffer,
                f"report_{machine_id}_{selected_cylinder_name}.pdf",
                "application/pdf",
                key='download_report'
            )
