"""
Health report table generation from XML metadata.

Streamlit is used for st.warning only (error display on failure).
"""

import xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd

from core.xml_parser import find_xml_value


def generate_health_report_table(source_xml_content, levels_xml_content, cylinder_index):
    try:
        source_root = ET.fromstring(source_xml_content)
        levels_root = ET.fromstring(levels_xml_content)
        col_idx = cylinder_index

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str:
                return "N/A"
            try:
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str

        def format_numeric_value(value_str, precision=2):
            if value_str == "N/A" or not value_str:
                return "N/A"
            try:
                return f"{float(value_str):.{precision}f}"
            except (ValueError, TypeError):
                return value_str

        suction_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2))
        discharge_p = convert_kpa_to_psi(find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2))
        suction_temp = find_xml_value(levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2)
        discharge_temp = find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx + 1)
        bore = find_xml_value(source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx + 1)
        rod_diam = find_xml_value(source_root, 'Source', 'PISTON ROD DIAMETER', col_idx + 1)

        # Extract raw values first
        comp_ratio_he_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=2)
        comp_ratio_ce_raw = find_xml_value(source_root, 'Source', 'COMPRESSION RATIO', col_idx + 1, occurrence=1)
        power_he_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=2)
        power_ce_raw = find_xml_value(source_root, 'Source', 'HORSEPOWER INDICATED,  LOAD', col_idx + 1, occurrence=1)

        # Apply formatting to the extracted values
        comp_ratio_he = format_numeric_value(comp_ratio_he_raw, precision=2)
        comp_ratio_ce = format_numeric_value(comp_ratio_ce_raw, precision=2)
        power_he = format_numeric_value(power_he_raw, precision=1)
        power_ce = format_numeric_value(power_ce_raw, precision=1)

        data = {
            'Cyl End': [f'{cylinder_index}H', f'{cylinder_index}C'],
            'Bore (ins)': [bore] * 2,
            'Rod Diam (ins)': ['N/A', rod_diam],
            'Pressure Ps/Pd (psig)': [f"{suction_p} / {discharge_p}"] * 2,
            'Temp Ts/Td (°C)': [f"{suction_temp} / {discharge_temp}"] * 2,
            'Comp. Ratio': [comp_ratio_he, comp_ratio_ce],
            'Indicated Power (ihp)': [power_he, power_ce]
        }
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Could not generate health report: {e}")
        return pd.DataFrame()
