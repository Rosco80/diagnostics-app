"""
XML parsing utilities: low-level XML value extraction, curve data loading,
and RPM extraction from Windrock XML export files.

Also provides load_wrpm_curves_data() as a WRPM adapter that returns the same
dict format as load_all_curves_data() so downstream analysis is format-agnostic.

No Streamlit imports — pure data logic only.
"""

import xml.etree.ElementTree as ET
import re
import pandas as pd
from io import BytesIO


def find_xml_value(root, sheet_name, partial_key, col_offset, occurrence=1):
    try:
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == sheet_name), None)
        if ws is None:
            return "N/A"
        rows = ws.findall('.//ss:Row', NS)
        match_count = 0
        for row in rows:
            all_cells_in_row = row.findall('ss:Cell', NS)
            if not all_cells_in_row:
                continue
            first_cell_data_node = all_cells_in_row[0].find('ss:Data', NS)
            if first_cell_data_node is None or first_cell_data_node.text is None:
                continue
            if partial_key.upper() in (first_cell_data_node.text or "").strip().upper():
                match_count += 1
                if match_count == occurrence:
                    target_idx = col_offset + 1
                    dense_cells = {}
                    current_idx = 1
                    for cell in all_cells_in_row:
                        ss_index_str = cell.get(f'{{{NS["ss"]}}}Index')
                        if ss_index_str:
                            current_idx = int(ss_index_str)
                        dense_cells[current_idx] = cell
                        current_idx += 1
                    if target_idx in dense_cells:
                        value_node = dense_cells[target_idx].find('ss:Data', NS)
                        return value_node.text if value_node is not None and value_node.text else "N/A"
                    return "N/A"
        return "N/A"
    except Exception:
        return "N/A"


def load_all_curves_data(curves_xml_content):
    """Load all curve data from the Curves XML file. Returns (df, column_names) or (None, None)."""
    try:
        root = ET.fromstring(curves_xml_content)
        NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        ws = next((ws for ws in root.findall('.//ss:Worksheet', NS) if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'), None)
        if ws is None:
            return None, None
        table = ws.find('.//ss:Table', NS)
        rows = table.findall('ss:Row', NS)
        header_cells = rows[1].findall('ss:Cell', NS)
        raw_headers = [c.find('ss:Data', NS).text or '' for c in header_cells]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]
        data = [[cell.find('ss:Data', NS).text for cell in r.findall('ss:Cell', NS)] for r in rows[6:]]
        if not data:
            return None, None
        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]
        # Use dropna(how='all') to only drop rows where ALL values are NaN
        # Also drop completely empty columns to clean up data
        df = pd.DataFrame(data, columns=actual_columns).apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna(axis=1, how='all')
        df.sort_values('Crank Angle', inplace=True)
        # Update actual_columns to match the remaining columns after cleanup
        actual_columns = df.columns.tolist()
        return df, actual_columns
    except Exception:
        return None, None


def extract_rpm(levels_xml_content):
    """Extract RPM from levels file."""
    try:
        levels_root = ET.fromstring(levels_xml_content)
        # Look for RPM in the levels file
        rpm = find_xml_value(levels_root, 'Levels', 'RPM', 2)
        if rpm and rpm != "N/A":
            try:
                rpm_val = float(rpm)
                if 100 <= rpm_val <= 10000:  # Reasonable RPM range
                    return f"{rpm_val:.0f}"
            except (ValueError, TypeError):
                pass
        return "N/A"
    except Exception:
        return "N/A"


def extract_rpm_from_source(source_xml_content):
    """
    Extract RPM directly from Source XML rows using Excel schema structure.
    """
    try:
        ns = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}
        root = ET.fromstring(source_xml_content)
        rows = root.findall(".//ss:Row", ns)

        for row in rows:
            cells = row.findall("ss:Cell", ns)
            if len(cells) >= 3:
                cell_value = cells[0].find("ss:Data", ns)
                if cell_value is not None and "RPM" in cell_value.text.upper():
                    rpm_data = cells[2].find("ss:Data", ns)
                    if rpm_data is not None:
                        try:
                            return f"{float(rpm_data.text):.0f}"
                        except (ValueError, TypeError):
                            return "N/A"
    except Exception as e:
        print("RPM extraction failed:", e)

    return "N/A"


def load_wrpm_curves_data(wrpm_bytes: bytes):
    """
    WRPM adapter: parse a .wrpm file and return data in the same format as
    load_all_curves_data() so all downstream analysis code is format-agnostic.

    Args:
        wrpm_bytes: raw bytes of the .wrpm file

    Returns:
        (curves_dict, machine_id, is_engine) where:
          - curves_dict: {'Crank Angle': [...], 'ColName': [...], ...}
                         (DataFrame.to_dict(orient='list') format)
          - machine_id: str — human-readable machine name from D6NAME3.DAT
          - is_engine: bool — True if the unit is an engine (should be blocked)

    Raises:
        Exception propagated from WrpmParserAE on parse failure.
    """
    # Import here to avoid circular imports and keep core/ free of heavy deps
    # at module load time (wrpm_parser_ae is in the repo root).
    import sys
    import os
    # Ensure the repo root is on sys.path so we can import from the top level.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from wrpm_parser_ae import WrpmParserAE

    buf = BytesIO(wrpm_bytes)
    parser = WrpmParserAE(buf)

    # Parse AE waveforms
    ae_df = parser.parse_to_dataframe()

    # Parse pressure waveforms (may be None)
    pressure_df = parser.parse_pressure_to_dataframe()

    machine_id = parser.machine_id or "Unknown"
    is_engine = parser.is_engine

    # Merge AE and pressure DataFrames on Crank Angle when both are available.
    # Both share the same crank angle grid (same session, same length), so a
    # simple concat on columns is safe — but we use merge to be robust against
    # any minor length mismatch and to avoid duplicate Crank Angle columns.
    if pressure_df is not None and not pressure_df.empty and len(pressure_df.columns) > 1:
        merged = ae_df.merge(
            pressure_df,
            on='Crank Angle',
            how='left',
            suffixes=('', '_pressure'),
        )
    else:
        merged = ae_df

    # Sort by crank angle to match XML parser output convention
    merged = merged.sort_values('Crank Angle').reset_index(drop=True)

    curves_dict = merged.to_dict(orient='list')
    return curves_dict, machine_id, is_engine
