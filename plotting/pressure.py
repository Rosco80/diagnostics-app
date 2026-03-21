"""
Pressure trace processing and application to Plotly figures.

Streamlit is used for sidebar feedback messages only.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go


def process_pressure_by_period(df, pressure_curve, period_selection, rpm=600):
    """
    Enhanced period processing with VERY visible differences.
    """
    if pressure_curve not in df.columns:
        return None

    pressure_data = df[pressure_curve].values.copy()

    try:
        if period_selection == "Median":
            # Simple median filter
            window = 21
            processed_pressure = np.array([
                np.median(pressure_data[max(0, i - window // 2):min(len(pressure_data), i + window // 2 + 1)])
                for i in range(len(pressure_data))
            ])

        elif period_selection == "Average":
            # Simple moving average
            window = 31
            processed_pressure = np.convolve(pressure_data, np.ones(window) / window, mode='same')

        elif period_selection == "Maximum":
            # Create upper envelope - VERY obvious difference
            window = 51
            processed_pressure = np.array([
                np.max(pressure_data[max(0, i - window // 2):min(len(pressure_data), i + window // 2 + 1)])
                for i in range(len(pressure_data))
            ])

        elif period_selection == "Minimum":
            # Create lower envelope - VERY obvious difference
            window = 51
            processed_pressure = np.array([
                np.min(pressure_data[max(0, i - window // 2):min(len(pressure_data), i + window // 2 + 1)])
                for i in range(len(pressure_data))
            ])

        elif period_selection == "Outer Envelope":
            # Amplify peaks significantly
            baseline = np.median(pressure_data)
            processed_pressure = baseline + (pressure_data - baseline) * 1.5

        elif period_selection == "Inner Envelope":
            # Compress towards median
            baseline = np.median(pressure_data)
            processed_pressure = baseline + (pressure_data - baseline) * 0.3

        elif period_selection == "All periods":
            # Return original data unchanged
            return pressure_data

        else:
            # Unknown selection - return original
            return pressure_data

        return processed_pressure

    except Exception as e:
        print(f"Period processing failed: {e}")
        return pressure_data


def validate_pressure_signals(df, cylinder_config, pressure_options):
    """
    HolizTech-style signal validation - checks actual CURVES data instead of LEVELS data.
    Returns a dict mapping signal name -> "✅" or "❌".
    """
    validation_results = {}

    # Check CE PT trace using cylinder_config (works for all cylinders)
    if pressure_options.get('show_ce_pt', False):
        # Get CE pressure curve from cylinder_config
        ce_pressure_curve = cylinder_config.get('ce_pressure_curve')

        if ce_pressure_curve and ce_pressure_curve in df.columns:
            ce_pressure_data = df[ce_pressure_curve]  # Use the actual time-series data

            # Quality checks on the actual time-series data
            has_data = len(ce_pressure_data) > 0
            no_all_zeros = not (ce_pressure_data == 0).all()
            has_variation = ce_pressure_data.std() > 1.0  # Some variation in the data
            reasonable_range = (ce_pressure_data.min() >= -500) and (ce_pressure_data.max() < 10000)  # Allow negative (suction)
            no_excessive_spikes = (ce_pressure_data.std() < 2000)  # Not too erratic

            is_valid = has_data and no_all_zeros and has_variation and reasonable_range and no_excessive_spikes
            validation_results['CE PT trace'] = "✅" if is_valid else "❌"
        else:
            validation_results['CE PT trace'] = "❌"  # No CE data found in time-series

    # Check HE PT trace using cylinder_config (works for all cylinders)
    if pressure_options.get('show_he_pt', False):
        # Get HE pressure curve from cylinder_config
        he_pressure_curve = cylinder_config.get('he_pressure_curve')

        if he_pressure_curve and he_pressure_curve in df.columns:
            he_pressure_data = df[he_pressure_curve]  # Use the actual time-series data

            # Same quality checks as CE but on actual time-series data
            has_data = len(he_pressure_data) > 0
            no_all_zeros = not (he_pressure_data == 0).all()
            has_variation = he_pressure_data.std() > 1.0  # Some variation in the data
            reasonable_range = (he_pressure_data.min() >= -500) and (he_pressure_data.max() < 10000)  # Allow negative (suction)
            no_excessive_spikes = (he_pressure_data.std() < 2000)  # Not too erratic

            is_valid = has_data and no_all_zeros and has_variation and reasonable_range and no_excessive_spikes
            validation_results['HE PT trace'] = "✅" if is_valid else "❌"
        else:
            validation_results['HE PT trace'] = "❌"  # No HE data found in time-series

    return validation_results


def apply_pressure_options_to_plot(fig, df, cylinder_config, pressure_options, files_content):
    """
    Apply pressure options - CORRECTED to preserve existing pressure line (black).
    """
    if not pressure_options['enable_pressure']:
        return fig

    # Extract cylinder name from cylinder_config for error messages
    cylinder_name = cylinder_config.get('cylinder_name', 'Cylinder 1')

    # Color scheme for different traces
    trace_colors = {
        'he_pt': 'blue',
        'ce_pt': 'orange',  # Changed to orange to avoid confusion with red theoretical
        'he_theoretical': 'darkgreen',
        'ce_theoretical': 'red',  # Keep as red (this works correctly)
        'he_nozzle': 'darkblue',
        'ce_nozzle': 'darkred',
        'he_terminal': 'navy',
        'ce_terminal': 'maroon'
    }

    # Show CE (Crank End) pressure trace - ADD to existing plot, don't replace
    if pressure_options['show_ce_pt']:
        # Use cylinder_config to get CE pressure curve directly
        ce_pressure_curve = cylinder_config.get('ce_pressure_curve')

        if ce_pressure_curve and ce_pressure_curve in df.columns:
            ce_pressure_col = ce_pressure_curve

            # Check if we already added this trace to avoid duplicates
            existing_ce_traces = [trace.name for trace in fig.data if trace.name and 'CE PT trace' in trace.name]

            if not existing_ce_traces:  # Only add if not already present
                # Apply period selection processing
                processed_pressure = process_pressure_by_period(df, ce_pressure_col, pressure_options.get('period_selection', 'Median'))

                if processed_pressure is not None:
                    trace_name = f"CE PT trace ({pressure_options.get('period_selection', 'Median')})"

                    fig.add_trace(
                        go.Scatter(
                            x=df['Crank Angle'],
                            y=processed_pressure,
                            name=trace_name,
                            line=dict(color=trace_colors['ce_pt'], width=2, dash='solid'),
                            mode='lines'
                        ),
                        secondary_y=False
                    )
                    st.sidebar.success(f"✅ Added {trace_name}")
                else:
                    st.sidebar.error("❌ CE pressure processing failed")
            else:
                st.sidebar.info("CE PT trace already exists")
        else:
            st.sidebar.error(f"❌ No CE pressure curve found for {cylinder_name}")

    # Show HE (Head End) pressure trace - ADD to existing plot, don't replace
    if pressure_options['show_he_pt']:
        # Use cylinder_config to get HE pressure curve directly
        he_pressure_curve = cylinder_config.get('he_pressure_curve')

        if he_pressure_curve and he_pressure_curve in df.columns:
            he_pressure_col = he_pressure_curve

            # Check if we already added this trace to avoid duplicates
            existing_he_traces = [trace.name for trace in fig.data if trace.name and 'HE PT trace' in trace.name]

            if not existing_he_traces:  # Only add if not already present
                # Apply period selection processing
                processed_he_pressure = process_pressure_by_period(df, he_pressure_col, pressure_options.get('period_selection', 'Median'))

                if processed_he_pressure is not None:
                    trace_name = f"HE PT trace ({pressure_options.get('period_selection', 'Median')})"

                    fig.add_trace(
                        go.Scatter(
                            x=df['Crank Angle'],
                            y=processed_he_pressure,
                            name=trace_name,
                            line=dict(color=trace_colors['he_pt'], width=2, dash='solid'),
                            mode='lines'
                        ),
                        secondary_y=False
                    )
                    st.sidebar.success(f"✅ Added {trace_name}")
                else:
                    st.sidebar.error("❌ HE pressure processing failed")
            else:
                st.sidebar.info("HE PT trace already exists")
        else:
            st.sidebar.error(f"❌ No HE pressure curve found for {cylinder_name}")

    return fig
