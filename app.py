"""
AI-Powered Machine Diagnostics Analyzer
Thin orchestration layer — imports and calls all modules.
"""

import re
import pandas as pd
import streamlit as st

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Machine Diagnostics Analyzer")

# --- INITIALIZE SESSION STATE ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'active_session_id' not in st.session_state:
    st.session_state.active_session_id = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'valve_event_tags' not in st.session_state:
    st.session_state.valve_event_tags = {}
if 'pending_tag' not in st.session_state:
    st.session_state.pending_tag = None
if 'vertical_offset' not in st.session_state:
    st.session_state.vertical_offset = 10.0

# --- Constants ---
TAG_FAULT_TYPES = [
    "Valve Opening", "Valve Leakage", "Valve Sticking",
    "Closing Hard or Slamming", "Broken or Missing Valve Parts",
    "Spring Fatigue or Failure", "Other",
]

# --- Module imports ---
from database.client import init_db
from database.queries import get_last_row_id, check_and_display_alerts, upsert_analysis_records
from core.xml_parser import load_all_curves_data, load_wrpm_curves_data
from core.config_discovery import auto_discover_configuration, get_all_cylinder_details, build_wrpm_config
from core.anomaly_detection import run_rule_based_diagnostics_enhanced
from core.health_scoring import compute_health_score
from plotting.cylinder_view import generate_cylinder_view
from plotting.pressure import validate_pressure_signals, apply_pressure_options_to_plot
from reporting.pdf import generate_pdf_report_enhanced
from reporting.tables import generate_health_report_table
from ui.components import (
    inject_custom_css, render_main_header,
    render_valve_sensors_table, render_all_cylinder_details, render_machine_info_bar,
)
from ui.file_upload import enhanced_file_upload_section
from ui.sidebar import (
    render_cylinder_selection_sidebar,
    render_pressure_options_sidebar,
    render_ai_model_tuning_section,
    render_valve_timing_config,
    render_pdf_export,
)
from ui.tagging import render_tagging_workflow
from ui.analytics import (
    render_labels_filter_sidebar,
    render_historical_trend,
    render_saved_labels,
    render_training_data_export,
    render_data_integrity_check,
)


# --- Streamlit-cached wrappers (keep @st.cache_data out of core/) ---
@st.cache_data
def _cached_load_all_curves_data(_curves_xml_content):
    return load_all_curves_data(_curves_xml_content)


@st.cache_data
def _cached_auto_discover_configuration(_source_xml_content, all_curve_names):
    return auto_discover_configuration(_source_xml_content, all_curve_names)


def main():
    db_client = init_db()
    inject_custom_css()
    render_main_header()

    # Apply pending WRPM view defaults (must happen BEFORE slider widgets are created)
    if '_wrpm_offset_pending' in st.session_state:
        st.session_state.vertical_offset = st.session_state.pop('_wrpm_offset_pending')

    # -------------------------------------------------------------------
    # Sidebar — file upload and view controls
    # -------------------------------------------------------------------
    with st.sidebar:
        validated_files = enhanced_file_upload_section()

        if st.session_state.analysis_results is not None:
            if st.button("Start New Analysis", type="secondary"):
                st.session_state.analysis_results = None
                st.session_state.active_session_id = None
                st.session_state.pop('auto_discover_config', None)
                st.cache_data.clear()
                st.rerun()

        st.header("2. View Options")
        envelope_view = st.checkbox("Enable Envelope View", value=True, key='envelope_view')
        vertical_offset = st.slider("Vertical Offset", 0.0, 50.0, step=1.0, key='vertical_offset')
        amplitude_scale = st.slider("Valve Amplitude Scale", 0.1, 5.0, 1.0, 0.1, key='amplitude_scale')
        dark_theme = st.checkbox("Black Background Theme", value=False, key='dark_theme')

        # Recommend offset based on valve count
        results = st.session_state.analysis_results
        if results and st.session_state.get('selected_cylinder_name'):
            cfg = results.get('discovered_config', {})
            sel_name = st.session_state.get('selected_cylinder_name', '')
            sel_cyl = next((c for c in cfg.get('cylinders', []) if c.get('cylinder_name') == sel_name), None)
            if sel_cyl:
                nv = len(sel_cyl.get('valve_vibration_curves', []))
                rec = max(10, nv * 3)
                if nv > 0 and vertical_offset < rec * 0.7:
                    st.info(f"Tip: With {nv} valves detected, try offset ~{rec}")

        view_mode = st.radio("View Mode", ["Crank-angle", "P-V"], index=0, key='view_mode')
        show_pv_overlay = False
        interactive_tagging = False
        if view_mode == "Crank-angle":
            show_pv_overlay = st.checkbox("Show P-V Overlay", value=False, key='pv_overlay')
            interactive_tagging = st.checkbox("Interactive Tagging Mode", value=False, key='interactive_tagging')

        pressure_options = render_pressure_options_sidebar()
        clearance_pct = st.number_input("Clearance (%)", 0.0, 20.0, 5.0, 0.5, key='clearance_pct')

        if 'auto_discover_config' in st.session_state:
            contamination_level, pressure_limit, valve_limit = render_ai_model_tuning_section(
                db_client, st.session_state['auto_discover_config'])
        else:
            contamination_level, pressure_limit, valve_limit = 0.05, 10, 5

    # -------------------------------------------------------------------
    # Main content area
    # -------------------------------------------------------------------
    if not validated_files:
        st.warning("Please upload your XML data files to begin analysis.")
    else:
        files_content = validated_files

        # ── WRPM path ──────────────────────────────────────────────────────────
        # WRPM files are blocked at upload time if is_engine=True, so if we reach
        # here with a WRPM dict the file is confirmed to be a compressor.
        if files_content.get('_wrpm'):
            if st.session_state.analysis_results is None:
                with st.spinner("Processing WRPM data..."):
                    curves_dict = files_content['curves_dict']
                    machine_id = files_content['machine_id']

                    # Reconstruct DataFrame and column names from the stored dict
                    df = pd.DataFrame(curves_dict)
                    curve_names = [c for c in df.columns.tolist() if c != 'Crank Angle']

                    # Build cylinder config directly from WRPM column names — no XML needed
                    discovered_config = build_wrpm_config(curve_names)
                    if discovered_config:
                        # machine_id from D6NAME3.DAT is authoritative
                        discovered_config['machine_id'] = machine_id
                        st.session_state['auto_discover_config'] = discovered_config
                        rpm = discovered_config.get('rated_rpm', 'N/A')
                        if st.session_state.active_session_id is None:
                            db_client.execute(
                                "INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)",
                                (machine_id, rpm))
                            st.session_state.active_session_id = get_last_row_id(db_client)
                            st.success(f"New analysis session #{st.session_state.active_session_id} created.")
                        # Provide empty-string placeholders for source/levels so downstream
                        # code that calls generate_health_report_table() and
                        # render_all_cylinder_details() doesn't crash — those functions
                        # handle empty strings gracefully via their own try/except guards.
                        wrpm_files_content = {
                            '_wrpm': True,
                            'curves': '',
                            'source': '',
                            'levels': '',
                            'curves_dict': curves_dict,
                            'machine_id': machine_id,
                        }
                        st.session_state.analysis_results = {
                            'df': df, 'discovered_config': discovered_config,
                            'files_content': wrpm_files_content,
                            'rpm': rpm, 'machine_id': machine_id,
                        }
                        # Queue WRPM-appropriate view defaults (applied before slider on next rerun)
                        max_valves = max(
                            len(c.get('valve_vibration_curves', []))
                            for c in discovered_config.get('cylinders', [{}])
                        )
                        st.session_state['_wrpm_offset_pending'] = float(max(10, max_valves * 3))
                        st.success("WRPM analysis complete!")
                    else:
                        st.error(
                            "Could not build cylinder configuration from WRPM column names. "
                            "Ensure the WRPM file contains properly named AE sensor channels "
                            "(e.g. 'Comp 1 H Ultra', 'Comp 1 H 1 S Val Ultra')."
                        )

        # ── XML path ───────────────────────────────────────────────────────────
        elif not ('curves' in files_content and 'source' in files_content and 'levels' in files_content):
            st.error("Failed to process curve data.")
        else:
            # Run heavy analysis once; cache results in session state
            if st.session_state.analysis_results is None:
                with st.spinner("Processing data..."):
                    df, curve_names = _cached_load_all_curves_data(files_content['curves'])
                    if df is not None:
                        discovered_config = _cached_auto_discover_configuration(
                            files_content['source'], curve_names)
                        if discovered_config:
                            st.session_state['auto_discover_config'] = discovered_config
                            rpm = discovered_config.get('rated_rpm', 'N/A')
                            machine_id = discovered_config.get('machine_id', 'N/A')
                            if st.session_state.active_session_id is None:
                                db_client.execute(
                                    "INSERT INTO sessions (machine_id, rpm) VALUES (?, ?)",
                                    (machine_id, rpm))
                                st.session_state.active_session_id = get_last_row_id(db_client)
                                st.success(f"New analysis session #{st.session_state.active_session_id} created.")
                            st.session_state.analysis_results = {
                                'df': df, 'discovered_config': discovered_config,
                                'files_content': files_content, 'rpm': rpm, 'machine_id': machine_id,
                            }
                            st.success("Analysis complete!")

        # ── Shared rendering (WRPM and XML both use this block) ───────────────
        if st.session_state.analysis_results:
            r = st.session_state.analysis_results
            df = r['df']
            discovered_config = r['discovered_config']
            files_content = r['files_content']
            rpm = r['rpm']
            machine_id = r['machine_id']
            cylinders = discovered_config.get("cylinders", [])

            with st.sidebar:
                selected_cylinder_name, selected_cylinder_config = render_cylinder_selection_sidebar(
                    discovered_config)
                if pressure_options['enable_pressure'] and selected_cylinder_config:
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### Signal Validation Status")
                    validation = validate_pressure_signals(df, selected_cylinder_config, pressure_options)
                    for sig, status in (validation or {}).items():
                        (st.sidebar.success if status == "OK" else st.sidebar.error)(f"{status} {sig}")
                    if not validation:
                        st.sidebar.info("No pressure signals selected for validation")

            if selected_cylinder_config:
                cylinder_index = int(re.search(r'\d+', selected_cylinder_name).group())

                # First pass: generate fig to build trace→column mapping
                fig, temp_report_data = generate_cylinder_view(
                    db_client, df.copy(), selected_cylinder_config,
                    envelope_view, vertical_offset, {}, contamination_level,
                    view_mode=view_mode, clearance_pct=clearance_pct,
                    show_pv_overlay=show_pv_overlay, amplitude_scale=amplitude_scale,
                    dark_theme=dark_theme)

                # Build display-name → actual column mapping for tagging click events
                mapping = {}
                for trace in fig.data:
                    if trace.name:
                        for item in temp_report_data:
                            if item.get('name', '') and item['name'] in trace.name:
                                mapping[trace.name] = item['curve_name']
                                break
                if 'curve_name_mapping' not in st.session_state:
                    st.session_state.curve_name_mapping = {}
                st.session_state.curve_name_mapping[selected_cylinder_name] = mapping

                if view_mode == "Crank-angle":
                    fig = apply_pressure_options_to_plot(
                        fig, df.copy(), selected_cylinder_config, pressure_options, files_content)

                if interactive_tagging and view_mode == "Crank-angle":
                    render_tagging_workflow(fig, df, temp_report_data, db_client,
                                            selected_cylinder_name, TAG_FAULT_TYPES)

                # Upsert DB records; regenerate plot with valve events
                analysis_ids = upsert_analysis_records(
                    db_client, st.session_state.active_session_id,
                    selected_cylinder_name, temp_report_data)

                fig, report_data = generate_cylinder_view(
                    db_client, df.copy(), selected_cylinder_config,
                    envelope_view, vertical_offset, analysis_ids, contamination_level,
                    view_mode=view_mode, clearance_pct=clearance_pct,
                    show_pv_overlay=show_pv_overlay, amplitude_scale=amplitude_scale,
                    dark_theme=dark_theme)

                if view_mode == "Crank-angle":
                    fig = apply_pressure_options_to_plot(
                        fig, df.copy(), selected_cylinder_config, pressure_options, files_content)

                if not (interactive_tagging and view_mode == "Crank-angle"):
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"updated_plot_{selected_cylinder_name}")

                # Diagnostics + health
                suggestions, critical_alerts = run_rule_based_diagnostics_enhanced(
                    report_data, pressure_limit, valve_limit)
                if suggestions:
                    st.subheader("Rule-Based Diagnostics")
                    for name, suggestion in suggestions.items():
                        st.warning(f"{name}: {suggestion}")

                health_score = compute_health_score(report_data, suggestions)
                check_and_display_alerts(db_client, machine_id, selected_cylinder_name,
                                         critical_alerts, health_score)
                st.metric("Health Score", f"{health_score:.1f}")

                # Health report, valve table, timing config, PDF export, machine info
                st.subheader("Compressor Health Report")
                health_report_df = generate_health_report_table(
                    files_content['source'], files_content['levels'], cylinder_index)
                if not health_report_df.empty:
                    st.dataframe(health_report_df, use_container_width=True, hide_index=True)

                render_valve_sensors_table(
                    selected_cylinder_config.get('valve_vibration_curves', []), cylinder_index)
                render_valve_timing_config(db_client, report_data, selected_cylinder_name)
                render_pdf_export(
                    db_client, machine_id, rpm, selected_cylinder_name,
                    report_data, health_report_df, fig, suggestions,
                    health_score, critical_alerts, generate_pdf_report_enhanced)

                st.markdown("---")
                render_machine_info_bar(discovered_config)
                render_all_cylinder_details(
                    get_all_cylinder_details(files_content['source'], files_content['levels'],
                                             len(cylinders)))

    # -------------------------------------------------------------------
    # Analytics (always rendered, filtered by machine ID from sidebar)
    # -------------------------------------------------------------------
    with st.sidebar:
        machine_id_filter = render_labels_filter_sidebar(db_client)

    render_historical_trend(db_client)
    render_saved_labels(db_client, machine_id_filter)
    render_training_data_export(db_client, machine_id_filter)
    render_data_integrity_check(db_client)


main()
