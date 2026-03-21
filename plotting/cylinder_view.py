"""
Cylinder diagnostic view generation.

Streamlit is used for informational/warning messages only.
The public entry point is generate_cylinder_view().
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.anomaly_detection import run_anomaly_detection


# ---------------------------------------------------------------------------
# Volume series helper (used by P-V diagram and overlay)
# ---------------------------------------------------------------------------

def _compute_volume_series(crank_angles, bore, stroke, clearance_pct):
    """
    Computes instantaneous cylinder volume for each crank angle for P-V diagram.
    FIXED: Proper index alignment to prevent out-of-bounds errors.
    """
    if len(crank_angles) == 0:
        return pd.Series([], dtype=float)

    try:
        # Convert inputs to float
        bore_f = float(bore)
        stroke_f = float(stroke)
        clearance_f = float(clearance_pct) / 100.0

        # Calculate swept volume
        area = math.pi * (bore_f / 2) ** 2
        swept_volume = area * stroke_f
        clearance_volume = swept_volume * clearance_f

        # Convert crank angles - preserve original index from DataFrame
        if isinstance(crank_angles, pd.Series):
            # If it's already a pandas Series, use it directly
            crank_angles_series = crank_angles.astype(float)
        else:
            # If it's a list/array, convert to Series with default index
            crank_angles_series = pd.Series(crank_angles, dtype=float)

        # Convert to radians
        theta_rad = np.deg2rad(crank_angles_series)

        # Calculate piston position using kinematic formula
        # For simplified MVP: piston_position = stroke/2 * (1 - cos(theta))
        piston_position = (stroke_f / 2) * (1 - np.cos(theta_rad))

        # Calculate instantaneous volume - preserve the original index
        instantaneous_volume = clearance_volume + area * piston_position

        # Return with the same index as the input crank_angles
        return pd.Series(instantaneous_volume.values, index=crank_angles_series.index)

    except (ValueError, TypeError) as e:
        st.warning(f"Volume computation error: {e}")
        return pd.Series([], dtype=float)

    except Exception as e:
        st.error(f"Volume computation error: {e}")
        return None


# ---------------------------------------------------------------------------
# Sub-function: P-V diagram (standalone)
# ---------------------------------------------------------------------------

def _render_pv_diagram(df, cylinder_config, bore, stroke, pressure_curve, clearance_pct, dark_theme):
    """Build and return (fig, report_data) for standalone P-V diagram view."""
    report_data = []

    try:
        V = _compute_volume_series(df["Crank Angle"], bore, stroke, clearance_pct)

        if V is not None and len(V) > 0:
            pressure_data = df[pressure_curve]

            if len(V) == len(pressure_data):
                fig = go.Figure()

                # Add the P-V cycle
                fig.add_trace(go.Scatter(
                    x=V, y=pressure_data,
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=2),
                    name="P-V Cycle",
                    hovertemplate="<b>Volume:</b> %{x:.1f} in³<br>" +
                                  "<b>Pressure:</b> %{y:.1f} PSIG<br>" +
                                  "<extra></extra>"
                ))

                try:
                    # STANDALONE P-V MODE: Find TDC/BDC points for P-V diagram
                    if len(V) > 0 and len(pressure_data) > 0 and len(V) == len(pressure_data):
                        # Use numpy arrays for safer operations
                        volume_values = V.values
                        pressure_values = pressure_data.values

                        # Find positions of min and max volume (filter NaN from BOTH volume and pressure)
                        vol_valid_mask = ~np.isnan(volume_values)
                        pressure_valid_mask = ~np.isnan(pressure_values)
                        combined_valid_mask = vol_valid_mask & pressure_valid_mask

                        if combined_valid_mask.sum() > 0:
                            valid_indices = np.where(combined_valid_mask)[0]
                            valid_volumes = volume_values[combined_valid_mask]
                            min_vol_pos = valid_indices[np.argmin(valid_volumes)]
                            max_vol_pos = valid_indices[np.argmax(valid_volumes)]
                        else:
                            min_vol_pos = 0
                            max_vol_pos = len(volume_values) - 1 if len(volume_values) > 0 else 0

                        # Get the actual values for P-V plot (Volume on X-axis)
                        min_vol = volume_values[min_vol_pos]
                        max_vol = volume_values[max_vol_pos]
                        min_pressure = pressure_values[min_vol_pos]
                        max_pressure = pressure_values[max_vol_pos]

                        # Add TDC point (minimum volume) on P-V diagram
                        fig.add_trace(go.Scatter(
                            x=[min_vol], y=[min_pressure],  # X=Volume, Y=Pressure
                            mode="markers",
                            marker=dict(size=12, color="red", symbol="circle"),
                            name="TDC (Top Dead Center)"
                        ))

                        # Add BDC point (maximum volume) on P-V diagram
                        fig.add_trace(go.Scatter(
                            x=[max_vol], y=[max_pressure],  # X=Volume, Y=Pressure
                            mode="markers",
                            marker=dict(size=12, color="blue", symbol="square"),
                            name="BDC (Bottom Dead Center)"
                        ))

                        # Add annotations on P-V diagram
                        fig.add_annotation(
                            x=min_vol, y=min_pressure,
                            text="TDC", showarrow=True, arrowhead=2, ax=20, ay=-20
                        )
                        fig.add_annotation(
                            x=max_vol, y=max_pressure,
                            text="BDC", showarrow=True, arrowhead=2, ax=-20, ay=20
                        )

                        # Debug info for P-V diagram
                        st.info(f"TDC at {min_vol:.1f} in³ ({min_pressure:.1f} PSIG), BDC at {max_vol:.1f} in³ ({max_pressure:.1f} PSIG)")
                    else:
                        st.warning("Volume and pressure data length mismatch or empty data")

                except Exception as e:
                    st.warning(f"Could not mark TDC/BDC points: {str(e)}")

                # Apply theme for P-V diagram
                if dark_theme:
                    fig.update_layout(
                        height=700,
                        title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')}",
                        template="plotly_dark",
                        plot_bgcolor='black',
                        paper_bgcolor='black',
                        font=dict(color='white'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white')),
                        showlegend=True,
                        xaxis=dict(gridcolor='#444444', zerolinecolor='#666666'),
                        yaxis=dict(gridcolor='#444444', zerolinecolor='#666666')
                    )
                    fig.update_xaxes(title_text="<b>Volume (in³)</b>", color='white')
                    fig.update_yaxes(title_text="<b>Pressure (PSIG)</b>", color='white')
                else:
                    fig.update_layout(
                        height=700,
                        title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')}",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        showlegend=True
                    )
                    fig.update_xaxes(title_text="<b>Volume (in³)</b>")
                    fig.update_yaxes(title_text="<b>Pressure (PSIG)</b>")

                # Add pressure data to report_data
                if pressure_curve in df.columns:
                    anomaly_count = int(df[f'{pressure_curve}_anom'].sum())
                    avg_score = df.loc[df[f'{pressure_curve}_anom'], f'{pressure_curve}_anom_score'].mean() if anomaly_count > 0 else 0.0
                    report_data.append({
                        "name": "Pressure",
                        "curve_name": pressure_curve,
                        "threshold": avg_score,
                        "count": anomaly_count,
                        "unit": "PSIG"
                    })

                return fig, report_data
            else:
                st.warning(f"Data length mismatch: Volume={len(V)}, Pressure={len(pressure_data)}")
        else:
            st.warning("Failed to compute volume data")

    except Exception as e:
        st.warning(f"P-V diagram computation failed: {e}")

    # Return empty figure if P-V plot fails
    fig = go.Figure()
    if dark_theme:
        fig.update_layout(
            height=700,
            title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')} (Error)",
            template="plotly_dark",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        fig.add_annotation(
            text="Unable to generate P-V diagram",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color='white', size=16)
        )
    else:
        fig.update_layout(
            height=700,
            title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')} (Error)",
            template="plotly_white"
        )
        fig.add_annotation(
            text="Unable to generate P-V diagram",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
    return fig, report_data


# ---------------------------------------------------------------------------
# Sub-function: valve annotations from DB
# ---------------------------------------------------------------------------

def _add_valve_annotations(fig, db_client, analysis_ids, curve_name, label_name, current_offset, color_rgba):
    """Add valve open/close event markers from the database to the figure."""
    if not analysis_ids:
        return fig

    # Get session context from first analysis
    first_analysis_id = list(analysis_ids.values())[0]
    try:
        context_rs = db_client.execute(
            "SELECT session_id, cylinder_name FROM analyses WHERE id = ?",
            (first_analysis_id,)
        )
        if context_rs.rows:
            session_id, cylinder_name = context_rs.rows[0]

            # Query valve events for this curve
            events_raw = db_client.execute(
                "SELECT curve_type, crank_angle FROM valve_events WHERE session_id = ? AND cylinder_name = ? AND curve_name = ?",
                (session_id, cylinder_name, curve_name)
            ).rows

            # Only process events if we have data
            if events_raw:
                events = {etype: angle for etype, angle in events_raw}
                # Clean visualization: colored triangles at valve's vertical position
                # Annotations shown on hover only
                for event_type, crank_angle in events.items():
                    # Use marker symbol instead of annotation
                    marker_symbol = "triangle-up" if event_type == 'open' else "triangle-down"
                    fig.add_trace(
                        go.Scatter(
                            x=[crank_angle],
                            y=[current_offset],  # Position at valve's vertical offset
                            mode='markers',
                            marker=dict(
                                symbol=marker_symbol,
                                size=18,  # Bigger markers
                                color=color_rgba.replace('0.4', '1'),
                                line=dict(width=2, color='white')
                            ),
                            name=f"{label_name} {event_type.capitalize()}",
                            showlegend=False,
                            hovertemplate=f"<b>{label_name}</b><br>{event_type.capitalize()}: {crank_angle}°<extra></extra>",
                        ),
                        secondary_y=True
                    )
        # No warning needed when no valve events exist - this is normal
    except Exception:
        pass  # Silently skip if valve events can't be loaded

    return fig


# ---------------------------------------------------------------------------
# Sub-function: crank-angle waveform view (builds the main fig)
# ---------------------------------------------------------------------------

def _render_crank_angle_plot(df, cylinder_config, envelope_view, vertical_offset, analysis_ids, db_client, amplitude_scale, dark_theme):
    """
    Build the crank-angle waveform figure with pressure + valve vibration traces.
    Returns (fig, report_data).
    """
    pressure_curve = cylinder_config.get('pressure_curve')
    is_wrpm_pressure = pressure_curve is not None and '.PVPT.' in str(pressure_curve)
    pressure_trace_label = "Pressure (PSI diff)" if is_wrpm_pressure else "Pressure (PSIG)"
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add pressure curve to crank-angle plot
    if pressure_curve and pressure_curve in df.columns:
        anomaly_count = int(df[f'{pressure_curve}_anom'].sum())
        avg_score = df.loc[df[f'{pressure_curve}_anom'], f'{pressure_curve}_anom_score'].mean() if anomaly_count > 0 else 0.0
        report_data.append({
            "name": "Pressure",
            "curve_name": pressure_curve,
            "threshold": avg_score,
            "count": anomaly_count,
            "unit": "PSIG"
        })
        # Set pressure line color based on theme
        pressure_color = 'white' if dark_theme else 'black'
        # Smooth WRPM pressure with a light median filter to reduce ADC noise
        pressure_y = df[pressure_curve].copy()
        if is_wrpm_pressure and len(pressure_y) > 5:
            from scipy.ndimage import median_filter
            pressure_y = pd.Series(median_filter(pressure_y.values, size=5), index=pressure_y.index)
        fig.add_trace(
            go.Scatter(
                x=df['Crank Angle'],
                y=pressure_y,
                name=pressure_trace_label,
                line=dict(color=pressure_color, width=2),
                customdata=[[pressure_curve]] * len(df)  # Store actual column name
            ),
            secondary_y=False
        )

    # Add valve vibration curves
    # Custom high-contrast, colorblind-friendly palette
    distinct_colors = [
        (0.12, 0.47, 0.71),  # Dark Blue
        (1.0, 0.50, 0.05),   # Orange
        (0.17, 0.63, 0.17),  # Green
        (0.84, 0.15, 0.16),  # Red
        (0.58, 0.40, 0.74),  # Purple
        (0.09, 0.75, 0.81),  # Cyan
        (0.89, 0.47, 0.76),  # Pink
        (0.74, 0.74, 0.13),  # Yellow-Green
        (0.55, 0.34, 0.29),  # Brown
        (0.84, 0.20, 0.65),  # Magenta
        (0.0, 0.55, 0.55),   # Teal
        (0.50, 0.50, 0.50),  # Gray
    ]
    # Cycle through colors if more valves than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(valve_curves))]
    current_offset = 0

    for i, vc in enumerate(valve_curves):
        curve_name, label_name = vc['curve'], vc['name']
        if curve_name not in df.columns:
            continue

        color_rgba = f'rgba({colors[i][0] * 255},{colors[i][1] * 255},{colors[i][2] * 255},0.4)'

        # Auto-normalize: scale each valve's signal so its waveform detail fills ~80% of its
        # allocated band. Use the 95th percentile (not max) to prevent outlier spikes from
        # crushing the visible detail — AE signals are very spiky (mean ~0.15G, max ~16G).
        signal_p95 = float(np.percentile(df[curve_name].dropna(), 95))
        if signal_p95 > 0 and vertical_offset > 0:
            band_height = vertical_offset * 0.8
            auto_scale = band_height / signal_p95
        else:
            auto_scale = 1.0
        effective_scale = auto_scale * amplitude_scale
        # Clip scaled values so extreme spikes don't overflow into adjacent valve bands
        half_band = vertical_offset * 0.95 if vertical_offset > 0 else 1e9

        color_solid = f'rgba({colors[i][0] * 255:.0f},{colors[i][1] * 255:.0f},{colors[i][2] * 255:.0f},1)'
        color_fill = f'rgba({colors[i][0] * 255:.0f},{colors[i][1] * 255:.0f},{colors[i][2] * 255:.0f},0.15)'

        if envelope_view:
            scaled = np.clip(df[curve_name] * effective_scale, -half_band, half_band)
            upper_bound = scaled + current_offset
            lower_bound = -scaled + current_offset
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=1, color=color_solid),
                    showlegend=False,
                    hoverinfo='none'
                ),
                secondary_y=True
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=lower_bound,
                    mode='lines',
                    line=dict(width=1, color=color_solid),
                    fill='tonexty',
                    fillcolor=color_fill,
                    name=label_name,
                    hoverinfo='none',
                    customdata=[[curve_name]] * len(df)  # Store actual column name
                ),
                secondary_y=True
            )
        else:
            vibration_data = np.clip(df[curve_name] * effective_scale, -half_band, half_band) + current_offset
            fig.add_trace(
                go.Scatter(
                    x=df['Crank Angle'],
                    y=vibration_data,
                    name=label_name,
                    mode='lines',
                    line=dict(color=color_solid, width=1),
                    customdata=[[curve_name]] * len(df)  # Store actual column name
                ),
                secondary_y=True
            )

        # Add anomalies with confidence coloring
        anomalies_df = df[df[f'{curve_name}_anom']]
        if not anomalies_df.empty:
            anomaly_vibration_data = np.clip(anomalies_df[curve_name] * effective_scale, -half_band, half_band) + current_offset
            fig.add_trace(
                go.Scatter(
                    x=anomalies_df['Crank Angle'],
                    y=anomaly_vibration_data,
                    mode='markers',
                    name=f"{label_name} Anomalies",
                    marker=dict(
                        color=anomalies_df[f'{curve_name}_anom_confidence'],  # use confidence
                        colorscale='Reds',
                        showscale=True,  # display color bar
                        colorbar=dict(title="Anomaly Confidence", x=1.05)  # side legend
                    ),
                    hoverinfo='text',
                    text=[
                        f"Confidence: {conf:.2f} | Level: {lvl}"
                        for conf, lvl in zip(
                            anomalies_df[f'{curve_name}_anom_confidence'],
                            anomalies_df[f'{curve_name}_anom_level']
                        )
                    ],
                    showlegend=False
                ),
                secondary_y=True
            )

        # Add valve events from DB
        if db_client is not None:
            fig = _add_valve_annotations(fig, db_client, analysis_ids, curve_name, label_name, current_offset, color_rgba)

        # Add anomaly data to report
        anomaly_count = int(df[f'{curve_name}_anom'].sum())
        avg_score = df.loc[df[f'{curve_name}_anom'], f'{curve_name}_anom_score'].mean() if anomaly_count > 0 else 0.0
        report_data.append({
            "name": vc['name'],
            "curve_name": curve_name,
            "threshold": avg_score,
            "count": anomaly_count,
            "unit": "G"
        })

        current_offset += vertical_offset

    return fig, report_data, current_offset


# ---------------------------------------------------------------------------
# Sub-function: P-V overlay on crank-angle chart
# ---------------------------------------------------------------------------

def _render_pv_overlay(fig, df, cylinder_config, bore, stroke, pressure_curve, clearance_pct, dark_theme):
    """Add P-V overlay traces and TDC/BDC markers to a crank-angle figure."""
    try:
        V = _compute_volume_series(df["Crank Angle"], bore, stroke, clearance_pct)

        if V is not None and len(V) > 0:
            pressure_data = df[pressure_curve]

            if len(V) == len(pressure_data):
                # Scale volume data to fit nicely on the plot
                pressure_range = pressure_data.max() - pressure_data.min()
                volume_range = V.max() - V.min()

                # Scale volume to use about 20% of the pressure range at the top
                volume_scaled = ((V - V.min()) / volume_range) * (pressure_range * 0.2) + pressure_data.max() + (pressure_range * 0.05)

                # Add P-V overlay as a line
                fig.add_trace(
                    go.Scatter(
                        x=df['Crank Angle'],
                        y=volume_scaled,
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dot'),
                        name='Volume (scaled)',
                        hovertemplate="<b>Crank Angle:</b> %{x:.1f}°<br>" +
                                      "<b>Volume:</b> %{customdata:.1f} in³<br>" +
                                      "<extra></extra>",
                        customdata=V,
                        opacity=0.7
                    ),
                    secondary_y=False
                )

                try:
                    # P-V OVERLAY MODE: Find TDC/BDC for crank-angle chart with overlay
                    if len(V) > 0 and len(pressure_data) > 0 and len(V) == len(pressure_data) and len(df) == len(V):
                        # Use numpy arrays for safer operations
                        volume_values = V.values
                        pressure_values = pressure_data.values
                        crank_angles = df['Crank Angle'].values

                        # FIXED: Use PRESSURE-based detection instead of volume-based
                        # Smooth pressure to avoid noise in peak detection
                        window_size = min(20, len(pressure_values) // 10)
                        if window_size > 3:
                            # Create smoothed pressure using pandas rolling mean
                            pressure_series = pd.Series(pressure_values)
                            smoothed_pressure = pressure_series.rolling(window=window_size, center=True, min_periods=1).mean().values
                        else:
                            smoothed_pressure = pressure_values

                        # TDC: Find where pressure is maximum (peak compression)
                        # Filter positions where BOTH crank_angles AND pressure are valid
                        valid_crank_mask = ~np.isnan(crank_angles)
                        valid_pressure_mask = ~np.isnan(smoothed_pressure)
                        valid_mask = valid_crank_mask & valid_pressure_mask  # Both must be valid

                        if valid_mask.sum() > 0:
                            valid_indices = np.where(valid_mask)[0]
                            valid_pressures = smoothed_pressure[valid_mask]
                            tdc_pos = valid_indices[np.argmax(valid_pressures)]
                        else:
                            tdc_pos = 0

                        # BDC: Find pressure minimum in first 60% of cycle (before peak compression)
                        search_end = int(len(pressure_values) * 0.6)
                        if search_end > 0:
                            search_pressures = smoothed_pressure[:search_end]
                            search_crank_angles = crank_angles[:search_end]
                            # Filter where BOTH crank angles AND pressure are valid
                            search_crank_valid = ~np.isnan(search_crank_angles)
                            search_pressure_valid = ~np.isnan(search_pressures)
                            search_valid_mask = search_crank_valid & search_pressure_valid

                            if search_valid_mask.sum() > 0:
                                search_valid_indices = np.where(search_valid_mask)[0]
                                search_valid_pressures = search_pressures[search_valid_mask]
                                bdc_pos = search_valid_indices[np.argmin(search_valid_pressures)]
                            else:
                                bdc_pos = 0
                        else:
                            bdc_pos = 0

                        # Get crank angles and pressures for markers
                        tdc_crank_angle = crank_angles[tdc_pos]  # X=Crank Angle
                        bdc_crank_angle = crank_angles[bdc_pos]  # X=Crank Angle
                        tdc_pressure = pressure_values[tdc_pos]  # Y=Pressure (use original, not smoothed)
                        bdc_pressure = pressure_values[bdc_pos]  # Y=Pressure (use original, not smoothed)

                        # Add TDC marker on crank-angle chart
                        fig.add_trace(
                            go.Scatter(
                                x=[tdc_crank_angle], y=[tdc_pressure],  # X=Crank Angle, Y=Pressure
                                mode='markers',
                                marker=dict(
                                    size=12, color='red', symbol='circle',
                                    line=dict(width=2, color='darkred')
                                ),
                                name='TDC',
                                hovertemplate="<b>TDC</b><br>Angle: %{x:.1f}°<br>Pressure: %{y:.1f} PSIG<extra></extra>"
                            ),
                            secondary_y=False
                        )

                        # Add BDC marker on crank-angle chart
                        fig.add_trace(
                            go.Scatter(
                                x=[bdc_crank_angle], y=[bdc_pressure],  # X=Crank Angle, Y=Pressure
                                mode='markers',
                                marker=dict(
                                    size=12, color='blue', symbol='square',
                                    line=dict(width=2, color='darkblue')
                                ),
                                name='BDC',
                                hovertemplate="<b>BDC</b><br>Angle: %{x:.1f}°<br>Pressure: %{y:.1f} PSIG<extra></extra>"
                            ),
                            secondary_y=False
                        )

                        # Add annotations on crank-angle chart
                        annotation_bg = "rgba(0,0,0,0.8)" if dark_theme else "rgba(255,255,255,0.8)"
                        fig.add_annotation(
                            x=tdc_crank_angle, y=tdc_pressure,
                            text="TDC", showarrow=True, arrowhead=2, ax=30, ay=-30,
                            bgcolor=annotation_bg, bordercolor="red",
                            font=dict(color="red", size=10)
                        )

                        fig.add_annotation(
                            x=bdc_crank_angle, y=bdc_pressure,
                            text="BDC", showarrow=True, arrowhead=2, ax=-30, ay=30,
                            bgcolor=annotation_bg, bordercolor="blue",
                            font=dict(color="blue", size=10)
                        )

                        # Add P-V overlay info box
                        fig.update_layout(
                            annotations=fig.layout.annotations + (dict(
                                x=0.02, y=0.98, xref="paper", yref="paper",
                                text=f"<b>P-V Overlay Active</b><br>Volume range: {V.min():.1f} - {V.max():.1f} in³<br>Clearance: {clearance_pct}%<br>TDC at {tdc_crank_angle:.1f}° | BDC at {bdc_crank_angle:.1f}°",
                                showarrow=False, bgcolor="rgba(128,0,128,0.1)",
                                bordercolor="purple", borderwidth=1,
                                font=dict(size=9), align="left"
                            ),)
                        )

                        st.info(f"P-V overlay active! TDC detected at {tdc_crank_angle:.1f}° (peak pressure), BDC at {bdc_crank_angle:.1f}° (min pressure).")
                    else:
                        st.warning("Data length mismatch between volume, pressure, and DataFrame")

                except Exception as e:
                    st.warning(f"Could not mark TDC/BDC points: {str(e)}")

            else:
                st.warning(f"P-V overlay failed: Data length mismatch (Volume={len(V)}, Pressure={len(pressure_data)})")
        else:
            st.warning("P-V overlay failed: Could not compute volume data")

    except Exception as e:
        st.warning(f"P-V overlay failed: {str(e)}")

    return fig


# ---------------------------------------------------------------------------
# Sub-function: dual view (side-by-side crank-angle + P-V)
# ---------------------------------------------------------------------------

def _render_dual_view(df, cylinder_config, envelope_view, vertical_offset, analysis_ids, db_client, contamination_level, clearance_pct, amplitude_scale, dark_theme):
    """
    Dual-view: crank-angle on the left, P-V on the right.
    Returns (fig, report_data) — currently delegates to _render_crank_angle_plot
    since the original app.py does not implement a true side-by-side layout and
    this path is not reachable in the original code (view_mode is "Crank-angle" or "P-V").
    Kept as an explicit hook for future extension without behaviour change.
    """
    return _render_crank_angle_plot(
        df, cylinder_config, envelope_view, vertical_offset,
        analysis_ids, db_client, amplitude_scale, dark_theme
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_cylinder_view(
    _db_client, df, cylinder_config, envelope_view, vertical_offset,
    analysis_ids, contamination_level, view_mode="Crank-angle",
    clearance_pct=5.0, show_pv_overlay=False, amplitude_scale=1.0,
    dark_theme=False, pressure_options=None
):
    """
    Generates cylinder view plots with pressure and valve vibration data.
    Public entry point — delegates to sub-functions based on view_mode.
    """
    pressure_curve = cylinder_config.get('pressure_curve')
    valve_curves = cylinder_config.get('valve_vibration_curves', [])
    report_data = []

    # Initialize variables needed for P-V calculations
    bore = cylinder_config.get("bore")
    stroke = cylinder_config.get("stroke")
    can_plot_pv = (
        (bore is not None) and (stroke is not None) and
        (pressure_curve is not None) and (pressure_curve in df.columns)
    )

    curves_to_analyze = [vc['curve'] for vc in valve_curves if vc['curve'] in df.columns]
    if pressure_curve and pressure_curve in df.columns:
        curves_to_analyze.append(pressure_curve)

    df = run_anomaly_detection(df, curves_to_analyze, contamination_level)

    # --- P-V mode (standalone) ---
    if view_mode == "P-V" and not show_pv_overlay:
        if can_plot_pv:
            return _render_pv_diagram(df, cylinder_config, bore, stroke, pressure_curve, clearance_pct, dark_theme)
        else:
            missing = []
            if bore is None:
                missing.append("bore dimension")
            if stroke is None:
                missing.append("stroke dimension")
            if pressure_curve is None or pressure_curve not in df.columns:
                missing.append("pressure curve")
            st.warning(f"P-V diagram not available - missing: {', '.join(missing)}")

        # Return empty figure
        fig = go.Figure()
        if dark_theme:
            fig.update_layout(
                height=700,
                title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')} (Error)",
                template="plotly_dark",
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )
            fig.add_annotation(
                text="Unable to generate P-V diagram",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color='white', size=16)
            )
        else:
            fig.update_layout(
                height=700,
                title_text=f"P-V Diagram — {cylinder_config.get('cylinder_name', 'Cylinder')} (Error)",
                template="plotly_white"
            )
            fig.add_annotation(
                text="Unable to generate P-V diagram",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        return fig, report_data

    # --- Crank-angle mode OR Dual view mode ---
    fig, report_data, current_offset = _render_crank_angle_plot(
        df, cylinder_config, envelope_view, vertical_offset,
        analysis_ids, _db_client, amplitude_scale, dark_theme
    )

    # Detect WRPM data by checking if pressure column uses .PVPT. suffix
    # (XML data uses COMPRESSOR PT / STATIC suffixes; WRPM uses .PVPT.)
    is_wrpm = pressure_curve is not None and '.PVPT.' in str(pressure_curve)
    pressure_label = "Pressure (PSI diff)" if is_wrpm else "Pressure (PSIG)"

    # Set up layout — scale chart height with valve count AND vertical offset
    # so the offset slider actually changes visual density
    n_valves = len(valve_curves)
    chart_height = max(700, int(n_valves * vertical_offset * 2.5)) if vertical_offset > 0 else 700
    title_suffix = " with P-V Overlay" if show_pv_overlay else ""
    # Apply dark theme or default theme
    if dark_theme:
        fig.update_layout(
            height=chart_height,
            title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}{title_suffix}",
            xaxis_title="Crank Angle (deg)",
            template="plotly_dark",
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='white')),
            xaxis=dict(gridcolor='#444444', zerolinecolor='#666666'),
            yaxis=dict(gridcolor='#444444', zerolinecolor='#666666'),
            yaxis2=dict(gridcolor='#444444', zerolinecolor='#666666')
        )
        fig.update_yaxes(title_text=f"<b>{pressure_label}</b>", color="white", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", color="cyan", secondary_y=True)
    else:
        fig.update_layout(
            height=chart_height,
            title_text=f"Diagnostics for {cylinder_config.get('cylinder_name', 'Cylinder')}{title_suffix}",
            xaxis_title="Crank Angle (deg)",
            template="ggplot2",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text=f"<b>{pressure_label}</b>", color="black", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vibration (G) with Offset</b>", color="blue", secondary_y=True)

    # For WRPM differential pressure: set Y-axis to actual data range with padding
    # (avoids the default 0-3000 PSIG scale which hides the differential waveform)
    if is_wrpm and pressure_curve and pressure_curve in df.columns:
        p_min = float(df[pressure_curve].min())
        p_max = float(df[pressure_curve].max())
        p_padding = (p_max - p_min) * 0.15
        fig.update_yaxes(range=[p_min - p_padding, p_max + p_padding], secondary_y=False)

    # Keep the crank-angle axis stable when annotations are added during tagging
    if view_mode == "Crank-angle":
        crank_angle_series = None
        if 'Crank Angle' in df.columns:
            crank_angle_series = df['Crank Angle']
        elif 'Crank_Angle' in df.columns:
            crank_angle_series = df['Crank_Angle']
        elif df.index.name in ('Crank Angle', 'Crank_Angle'):
            crank_angle_series = df.index.to_series()

        if crank_angle_series is not None:
            valid_angles = pd.Series(crank_angle_series).dropna()
            if not valid_angles.empty:
                min_angle = float(valid_angles.min())
                max_angle = float(valid_angles.max())
                if math.isfinite(min_angle) and math.isfinite(max_angle):
                    # Ensure a non-zero span even if all values are identical
                    if math.isclose(min_angle, max_angle):
                        padding = max(1.0, abs(min_angle) * 0.01 + 1.0)
                        min_angle -= padding
                        max_angle += padding
                    fig.update_xaxes(range=[min_angle, max_angle], autorange=False)

    # Dynamic Y-axis range for valves based on offset and valve count
    if len(valve_curves) > 0:
        # Calculate total offset range needed
        total_offset_range = len(valve_curves) * vertical_offset
        # Add 20% padding above and below
        y_max = total_offset_range * 1.2
        y_min = -total_offset_range * 0.2
        # Apply the calculated range to the secondary Y-axis (valves)
        fig.update_yaxes(range=[y_min, y_max], secondary_y=True)

    # --- ADD P-V OVERLAY if requested ---
    if show_pv_overlay and view_mode == "Crank-angle" and can_plot_pv:
        fig = _render_pv_overlay(fig, df, cylinder_config, bore, stroke, pressure_curve, clearance_pct, dark_theme)

    return fig, report_data
