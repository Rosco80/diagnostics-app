"""
Interactive tagging workflow: click-to-tag, fault classification form,
save/clear controls, and waveform persistence for ML training.
"""

import streamlit as st

from database.queries import save_anomaly_tag_to_db, save_waveform_data_to_db, get_last_row_id


def render_tagging_workflow(
    fig,
    df,
    temp_report_data,
    db_client,
    selected_cylinder_name,
    tag_fault_types,
):
    """
    Render the full interactive tagging UI on top of `fig`.

    Displays the plotly chart with point-selection enabled, handles the
    pending-tag classification form, and provides Save/Clear buttons that
    write classified tags and waveform data to the database.

    Returns the (possibly annotation-enriched) figure so callers can reuse it.
    """
    st.markdown("### Interactive Tagging Mode")
    st.info("Click on the curves to tag crank-angle positions where anomalies are suspected.")

    plot_key = f"{selected_cylinder_name.replace(' ', '_')}_plot"
    existing_tags = st.session_state.valve_event_tags.get(plot_key, [])

    # Draw existing tags on the figure
    for tag in existing_tags:
        if isinstance(tag, dict):
            angle = tag['angle']
            fault_type = tag['fault_classification']
            curve_name = tag.get('curve_name', '')
            if curve_name and curve_name != 'Unknown':
                annotation_text = f"{curve_name}: {fault_type} @ {angle:.1f}°"
            else:
                annotation_text = f"{fault_type}: {angle:.1f}°"
        else:
            angle = tag
            annotation_text = f"Tagged: {angle:.1f}°"
        fig.add_vline(
            x=angle, line_dash="dash", line_color="red", line_width=2,
            annotation_text=annotation_text, annotation_position="top"
        )

    # Pending-tag classification form
    if st.session_state.pending_tag is not None:
        st.markdown("#### Classify Your Tag")
        if isinstance(st.session_state.pending_tag, dict):
            pending_angle = st.session_state.pending_tag['angle']
            pending_curve = st.session_state.pending_tag.get('curve_name', 'Unknown')
            st.info(f"You clicked **{pending_curve}** at crank angle: **{pending_angle:.2f}°**")
        else:
            pending_angle = st.session_state.pending_tag
            pending_curve = 'Unknown'
            st.info(f"You clicked at crank angle: **{pending_angle:.2f}°**")

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_fault_type = st.selectbox(
                "What type of fault/anomaly did you observe?",
                tag_fault_types,
                index=0,
                key="fault_classification_select"
            )
        with col2:
            if st.button("Add Tag", key="confirm_tag"):
                if plot_key not in st.session_state.valve_event_tags:
                    st.session_state.valve_event_tags[plot_key] = []
                new_tag = {
                    'angle': pending_angle,
                    'fault_classification': selected_fault_type,
                    'curve_name': pending_curve
                }
                st.session_state.valve_event_tags[plot_key].append(new_tag)
                st.session_state.pending_tag = None
                st.success(f"Tagged **{pending_curve}** as '{selected_fault_type}' at {pending_angle:.2f}°")
                st.rerun()

            if st.button("Cancel", key="cancel_tag"):
                st.session_state.pending_tag = None
                st.rerun()

    # Interactive chart with click detection
    with st.container():
        st.markdown('<div class="full-width-plot">', unsafe_allow_html=True)
        clicked_data = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"interactive_chart_{plot_key}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Handle click events
    if hasattr(clicked_data, 'selection') and clicked_data.selection:
        if clicked_data.selection.get('points'):
            for point in clicked_data.selection['points']:
                clicked_x = point.get('x')
                curve_number = point.get('curve_number')

                display_name = None
                if curve_number is not None and curve_number < len(fig.data):
                    display_name = fig.data[curve_number].name

                # Resolve display name to actual column name via session-state mapping
                actual_curve_name = display_name
                if display_name and selected_cylinder_name in st.session_state.get('curve_name_mapping', {}):
                    mapping = st.session_state.curve_name_mapping[selected_cylinder_name]
                    actual_curve_name = mapping.get(display_name, display_name)

                if clicked_x is not None and st.session_state.pending_tag is None and actual_curve_name:
                    st.session_state.pending_tag = {
                        'angle': clicked_x,
                        'curve_name': actual_curve_name
                    }
                    st.rerun()

    # Current tags list + save/clear
    if existing_tags:
        st.markdown("#### Current Tags")
        for tag in existing_tags:
            if isinstance(tag, dict):
                curve_name = tag.get('curve_name', 'Unknown curve')
                display_name = curve_name
                if len(curve_name) > 50:
                    parts = curve_name.split('.')
                    if len(parts) >= 2:
                        display_name = f"{parts[1]}... ({parts[0]})"
                    else:
                        display_name = curve_name[:50] + "..."
                st.write(f"- **{display_name}**: {tag['fault_classification']} at {tag['angle']:.2f}°")
            else:
                st.write(f"- Legacy tag: {tag:.2f}°")

        action_cols = st.columns([1, 1])
        with action_cols[0]:
            if st.button("Save Tags", key="save_tags"):
                saved_count = 0
                waveform_count = 0

                for item in temp_report_data:
                    rs = db_client.execute(
                        "SELECT id FROM analyses WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
                        (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'])
                    )
                    existing_id_row = rs.rows[0] if rs.rows else None
                    if existing_id_row:
                        analysis_id = existing_id_row[0]
                    else:
                        db_client.execute(
                            "INSERT INTO analyses (session_id, cylinder_name, curve_name, anomaly_count, threshold) VALUES (?, ?, ?, ?, ?)",
                            (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], item['count'], item['threshold'])
                        )
                        analysis_id = get_last_row_id(db_client)

                    db_client.execute(
                        "DELETE FROM anomaly_tags WHERE session_id = ? AND cylinder_name = ? AND curve_name = ? AND tag_type = ?",
                        (st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], 'Manual Tag')
                    )
                    for tag in existing_tags:
                        if isinstance(tag, dict):
                            tag_curve = tag.get('curve_name', '')
                            item_curve = item['curve_name']
                            if tag_curve == item_curve:
                                save_anomaly_tag_to_db(db_client, st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], tag['angle'], tag['fault_classification'], 'Manual Tag')
                                saved_count += 1
                            elif tag_curve and item_curve and (tag_curve in item_curve or item_curve in tag_curve):
                                save_anomaly_tag_to_db(db_client, st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], tag['angle'], tag['fault_classification'], 'Manual Tag')
                                saved_count += 1
                        else:
                            save_anomaly_tag_to_db(db_client, st.session_state.active_session_id, selected_cylinder_name, item['curve_name'], tag, 'Legacy tag', 'Manual Tag')
                            saved_count += 1

                    # Save waveform data for ML training
                    curve_col = item['curve_name']
                    if curve_col in df.columns:
                        curve_type = "pressure" if item['name'] == "Pressure" or item['unit'] == "PSIG" else "vibration"
                        if 'Crank_Angle' in df.columns:
                            crank_angles = df['Crank_Angle'].values
                        elif df.index.name == 'Crank_Angle':
                            crank_angles = df.index.values
                        else:
                            crank_angles = df.index.values
                        save_waveform_data_to_db(
                            db_client,
                            st.session_state.active_session_id,
                            selected_cylinder_name,
                            curve_col,
                            crank_angles,
                            df[curve_col].values,
                            curve_type
                        )
                        waveform_count += 1

                st.success(f"Saved {saved_count} classified tags and {waveform_count} waveform datasets to database!")

        with action_cols[1]:
            if st.button("Clear Tags", key="clear_tags"):
                st.session_state.valve_event_tags[plot_key] = []
                st.rerun()

    return fig
