"""
Configuration auto-discovery from Windrock Source XML.

Also provides build_wrpm_config() which builds the same config dict from
WRPM DataFrame column names when no Source XML is available.

No Streamlit imports — pure data logic only.
"""

import re
import xml.etree.ElementTree as ET
import math

from core.xml_parser import find_xml_value


def auto_discover_configuration(source_xml_content, all_curve_names):
    try:
        source_root = ET.fromstring(source_xml_content)

        # Determine number of cylinders
        num_cyl_str = find_xml_value(
            source_root, 'Source', "COMPRESSOR NUMBER OF CYLINDERS", 2
        )
        if num_cyl_str == "N/A" or int(num_cyl_str) == 0:
            return None
        num_cylinders = int(num_cyl_str)

        # Machine identifier
        machine_id = find_xml_value(source_root, 'Source', "Machine", 1)

        # Additional machine-level metadata
        machine_model = find_xml_value(
            source_root, 'Source', 'COMPRESSOR MODEL', 2
        )
        serial_number = find_xml_value(
            source_root, 'Source', 'COMPRESSOR SERIAL NUMBER', 2
        )
        rated_rpm = find_xml_value(
            source_root, 'Source', 'COMPRESSOR RATED RPM', 2
        )
        rated_hp = find_xml_value(
            source_root, 'Source', 'COMPRESSOR RATED HP', 2
        )

        # Cylinder-specific metadata
        bore_values = []
        rod_diameter_values = []
        stroke_values = []
        num_cols_offset_start = 2  # first cylinder's value is in column index 2

        for cyl_idx in range(num_cylinders):
            bore = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER BORE',
                num_cols_offset_start + cyl_idx,
            )
            bore_values.append(float(bore) if bore not in [None, '', 'N/A'] else None)

            rod_dia = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER PISTON ROD DIAMETER',
                num_cols_offset_start + cyl_idx,
            )
            rod_diameter_values.append(float(rod_dia) if rod_dia not in [None, '', 'N/A'] else None)

            stroke = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR THROW STROKE LENGTH',
                num_cols_offset_start + cyl_idx,
            )
            stroke_values.append(float(stroke) if stroke not in [None, '', 'N/A'] else None)

        # Build configuration for each cylinder
        cylinders_config = []

        for i in range(1, num_cylinders + 1):

            # More robust pressure curve detection
            pressure_curve = None

            # Try Head End first (.{i}H.)
            he_pressure = next(
                (c for c in all_curve_names if f".{i}H." in c and ("STATIC" in c or "SPECIAL" in c) and "COMPRESSOR PT" in c),
                None
            )

            # Try Crank End (.{i}C.)
            ce_pressure = next(
                (c for c in all_curve_names if f".{i}C." in c and ("STATIC" in c or "SPECIAL" in c) and "COMPRESSOR PT" in c),
                None
            )

            # Store both HE and CE pressure curves (for dual trace support)
            # Keep legacy pressure_curve for backward compatibility (prefer HE, fall back to CE)
            pressure_curve = he_pressure or ce_pressure

            # Detect ALL valves (not just valve #1)
            valve_curves = []

            # Head End Discharge (.{i}HD1, .{i}HD2, .{i}HD3, ...)
            he_discharge_valves = [
                c for c in all_curve_names
                if f".{i}HD" in c and ("VIBRATION" in c or "ULTRASONIC" in c)
            ]
            for valve_curve in he_discharge_valves:
                # Extract valve number (e.g., HD1 -> 1, HD2 -> 2)
                valve_num = ""
                for char in valve_curve.split(f".{i}HD")[1]:
                    if char.isdigit():
                        valve_num += char
                    else:
                        break

                # Add sensor type suffix to differentiate VIBRATION vs ULTRASONIC
                if "ULTRASONIC" in valve_curve.upper():
                    sensor_type = " (US)"
                elif "VIBRATION" in valve_curve.upper():
                    sensor_type = " (VIB)"
                else:
                    sensor_type = ""

                valve_name = f"HE Discharge {valve_num}{sensor_type}" if valve_num else f"HE Discharge{sensor_type}"
                valve_curves.append({"name": valve_name, "curve": valve_curve})

            # Head End Suction (.{i}HS1, .{i}HS2, .{i}HS3, ...)
            he_suction_valves = [
                c for c in all_curve_names
                if f".{i}HS" in c and ("VIBRATION" in c or "ULTRASONIC" in c)
            ]
            for valve_curve in he_suction_valves:
                valve_num = ""
                for char in valve_curve.split(f".{i}HS")[1]:
                    if char.isdigit():
                        valve_num += char
                    else:
                        break

                # Add sensor type suffix to differentiate VIBRATION vs ULTRASONIC
                if "ULTRASONIC" in valve_curve.upper():
                    sensor_type = " (US)"
                elif "VIBRATION" in valve_curve.upper():
                    sensor_type = " (VIB)"
                else:
                    sensor_type = ""

                valve_name = f"HE Suction {valve_num}{sensor_type}" if valve_num else f"HE Suction{sensor_type}"
                valve_curves.append({"name": valve_name, "curve": valve_curve})

            # Crank End Discharge (.{i}CD1, .{i}CD2, .{i}CD3, ...)
            ce_discharge_valves = [
                c for c in all_curve_names
                if f".{i}CD" in c and ("VIBRATION" in c or "ULTRASONIC" in c)
            ]
            for valve_curve in ce_discharge_valves:
                valve_num = ""
                for char in valve_curve.split(f".{i}CD")[1]:
                    if char.isdigit():
                        valve_num += char
                    else:
                        break

                # Add sensor type suffix to differentiate VIBRATION vs ULTRASONIC
                if "ULTRASONIC" in valve_curve.upper():
                    sensor_type = " (US)"
                elif "VIBRATION" in valve_curve.upper():
                    sensor_type = " (VIB)"
                else:
                    sensor_type = ""

                valve_name = f"CE Discharge {valve_num}{sensor_type}" if valve_num else f"CE Discharge{sensor_type}"
                valve_curves.append({"name": valve_name, "curve": valve_curve})

            # Crank End Suction (.{i}CS1, .{i}CS2, .{i}CS3, ...)
            ce_suction_valves = [
                c for c in all_curve_names
                if f".{i}CS" in c and ("VIBRATION" in c or "ULTRASONIC" in c)
            ]
            for valve_curve in ce_suction_valves:
                valve_num = ""
                for char in valve_curve.split(f".{i}CS")[1]:
                    if char.isdigit():
                        valve_num += char
                    else:
                        break

                # Add sensor type suffix to differentiate VIBRATION vs ULTRASONIC
                if "ULTRASONIC" in valve_curve.upper():
                    sensor_type = " (US)"
                elif "VIBRATION" in valve_curve.upper():
                    sensor_type = " (VIB)"
                else:
                    sensor_type = ""

                valve_name = f"CE Suction {valve_num}{sensor_type}" if valve_num else f"CE Suction{sensor_type}"
                valve_curves.append({"name": valve_name, "curve": valve_curve})

            # More lenient condition - include cylinder if it has EITHER pressure OR valve data
            # This ensures we don't skip cylinders that might have partial data
            has_pressure = pressure_curve is not None
            has_valves = len(valve_curves) > 0

            if has_pressure or has_valves:
                bore = bore_values[i - 1] if i <= len(bore_values) else None
                rod_dia = rod_diameter_values[i - 1] if i <= len(rod_diameter_values) else None
                stroke = stroke_values[i - 1] if i <= len(stroke_values) else None
                volume = None
                if bore is not None and stroke is not None:
                    volume = math.pi * (bore / 2) ** 2 * stroke

                cylinder_config = {
                    "cylinder_name": f"Cylinder {i}",
                    "pressure_curve": pressure_curve,  # Legacy field (for backward compatibility)
                    "he_pressure_curve": he_pressure,  # Head End pressure curve
                    "ce_pressure_curve": ce_pressure,  # Crank End pressure curve
                    "valve_vibration_curves": valve_curves,
                    "bore": bore,
                    "rod_diameter": rod_dia,
                    "stroke": stroke,
                    "volume": volume,
                }

                cylinders_config.append(cylinder_config)
            # If no pressure or valve data found, skip this cylinder

        # Ensure cylinders are sorted by number to guarantee Cylinder 1 comes first
        cylinders_config.sort(key=lambda x: int(x['cylinder_name'].split()[-1]))

        if len(cylinders_config) == 0:
            return None

        return {
            "machine_id": machine_id,
            "model": machine_model,
            "serial_number": serial_number,
            "rated_rpm": rated_rpm,
            "rated_hp": rated_hp,
            "num_cylinders": num_cylinders,
            "cylinders": cylinders_config,
        }

    except Exception:
        return None


def get_all_cylinder_details(source_xml_content, levels_xml_content, num_cylinders):
    details = []
    try:
        source_root = ET.fromstring(source_xml_content)
        levels_root = ET.fromstring(levels_xml_content)

        def convert_kpa_to_psi(kpa_str):
            if kpa_str == "N/A" or not kpa_str:
                return "N/A"
            try:
                return f"{float(kpa_str) * 0.145038:.1f}"
            except (ValueError, TypeError):
                return kpa_str

        def format_flow_balance(value_str):
            if value_str == "N/A" or not value_str:
                return "N/A"
            try:
                return f"{float(value_str) * 100:.1f} %"
            except (ValueError, TypeError):
                return value_str

        stage_suction_p_psi = convert_kpa_to_psi(
            find_xml_value(levels_root, 'Levels', 'SUCTION PRESSURE GAUGE', 2)
        )
        stage_discharge_p_psi = convert_kpa_to_psi(
            find_xml_value(levels_root, 'Levels', 'DISCHARGE PRESSURE GAUGE', 2)
        )
        stage_suction_temp = find_xml_value(
            levels_root, 'Levels', 'SUCTION GAUGE TEMPERATURE', 2
        )

        for i in range(1, num_cylinders + 1):
            col_idx = i + 1

            # Retrieve flow balance values
            fb_ce_raw = find_xml_value(
                source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=1
            )
            fb_he_raw = find_xml_value(
                source_root, 'Source', 'FLOW BALANCE', col_idx, occurrence=2
            )

            # Extract dimension values
            bore_value = find_xml_value(
                source_root, 'Source', 'COMPRESSOR CYLINDER BORE', col_idx
            )
            rod_dia_value = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR CYLINDER PISTON ROD DIAMETER',
                col_idx,
            )
            stroke_value = find_xml_value(
                source_root,
                'Source',
                'COMPRESSOR THROW STROKE LENGTH',
                col_idx,
            )

            # Compute volume (in³) if bore and stroke are numeric
            volume_val = "N/A"
            try:
                if bore_value not in [None, '', 'N/A'] and stroke_value not in [
                    None,
                    '',
                    'N/A',
                ]:
                    bore_f = float(bore_value)
                    stroke_f = float(stroke_value)
                    volume_val = f"{math.pi * (bore_f / 2) ** 2 * stroke_f:.2f}"
            except Exception:
                volume_val = "N/A"

            detail = {
                "name": f"Cylinder {i}",
                "bore": f"{bore_value} in",
                "rod_diameter": f"{rod_dia_value} in"
                if rod_dia_value not in [None, '', 'N/A']
                else "N/A",
                "stroke": f"{stroke_value} in"
                if stroke_value not in [None, '', 'N/A']
                else "N/A",
                "volume": f"{volume_val} in³" if volume_val != "N/A" else "N/A",
                "suction_temp": f"{stage_suction_temp} °C",
                "discharge_temp": f"{find_xml_value(levels_root, 'Levels', 'COMP CYL, DISCHARGE TEMPERATURE', col_idx)} °C",
                "suction_pressure": f"{stage_suction_p_psi} psig",
                "discharge_pressure": f"{stage_discharge_p_psi} psig",
                "flow_balance_ce": format_flow_balance(fb_ce_raw),
                "flow_balance_he": format_flow_balance(fb_he_raw),
            }
            details.append(detail)

        return details
    except Exception:
        return []


def build_wrpm_config(curve_names: list) -> dict:
    """
    Build a config dict (same schema as auto_discover_configuration) from
    WRPM DataFrame column names — used when no Source XML is available.

    Column naming convention (full DataFrame key):
        "{machine_id} - Comp {N} {end} [valve_part].{suffix}.{sensor_id}"

    Where:
        {end}        = H (Head End) | C (Crank End)
        [valve_part] = "{M} S Val Ultra"  → Suction valve M
                     | "{M} D Val Ultra"  → Discharge valve M
                     | "Ultra"            → General AE (no specific valve)
        {suffix}     = "ULTRASONIC G 36KHZ - 44KHZ" or "PVPT"

    Pressure columns contain "Pressure" in the sensor segment and use ".PVPT."
    as suffix.  They are assigned to he_pressure_curve / ce_pressure_curve but
    are NOT listed in valve_vibration_curves.

    Returns the same schema as auto_discover_configuration():
        {
            "machine_id": str,
            "model": "N/A",
            "serial_number": "N/A",
            "rated_rpm": "N/A",
            "rated_hp": "N/A",
            "num_cylinders": int,
            "cylinders": [
                {
                    "cylinder_name": "Cylinder N",
                    "pressure_curve": str | None,
                    "he_pressure_curve": str | None,
                    "ce_pressure_curve": str | None,
                    "valve_vibration_curves": [
                        {"name": str, "curve": str},
                        ...
                    ],
                    "bore": None,
                    "rod_diameter": None,
                    "stroke": None,
                    "volume": None,
                },
                ...
            ],
        }

    Returns None if no cylinder data can be parsed from curve_names.
    """
    # Patterns that match the sensor segment (the part between " - " and ".{suffix}")
    # e.g. "Comp 1 H 3 S Val Ultra" or "Comp 2 C Pressure" or "Comp 1 H Ultra"
    _VALVE_RE = re.compile(
        r'Comp\s+(\d+)\s+'     # group 1: cylinder number N
        r'(H|C)\s+'            # group 2: H or C (head/crank end)
        r'(\d+)\s+'            # group 3: valve number M
        r'(S|D)\s+Val\s+Ultra' # group 4: S (suction) or D (discharge)
    )
    _GENERAL_AE_RE = re.compile(
        r'Comp\s+(\d+)\s+'     # group 1: cylinder number N
        r'(H|C)\s+'            # group 2: H or C
        r'Ultra'               # general AE, no specific valve
    )
    _PRESSURE_RE = re.compile(
        r'Comp\s+(\d+)\s+'     # group 1: cylinder number N
        r'(H|C)\s+'            # group 2: H or C
        r'Pressure'
    )

    def _strip_to_sensor_segment(col: str) -> str:
        """
        Strip machine-id prefix and .suffix.sensor_id suffix to expose the
        sensor segment, e.g. "Comp 1 H 3 S Val Ultra".

        The machine prefix ends at the last ' - ' before 'Comp'.
        The suffix starts at the first '.' after the sensor segment.
        """
        # Find the last occurrence of ' - Comp' to strip machine prefix
        idx = col.rfind(' - Comp')
        if idx == -1:
            return col
        seg = col[idx + 3:]  # drop ' - ', keep 'Comp ...'
        # Drop everything from the first '.' onward (the .ULTRASONIC / .PVPT suffix)
        dot_idx = seg.find('.')
        if dot_idx != -1:
            seg = seg[:dot_idx]
        return seg.strip()

    # Accumulate per-cylinder data
    # cyl_data[N] = {
    #   'he_pressure': col | None,
    #   'ce_pressure': col | None,
    #   'valves': [{'name': ..., 'curve': col}, ...]
    # }
    cyl_data: dict = {}

    machine_id = "Unknown"
    for col in curve_names:
        seg = _strip_to_sensor_segment(col)

        # Try to extract machine_id from the first column that contains ' - Comp'
        if machine_id == "Unknown":
            idx = col.rfind(' - Comp')
            if idx != -1:
                machine_id = col[:idx].strip()

        # Pressure column
        m = _PRESSURE_RE.match(seg)
        if m:
            cyl_n = int(m.group(1))
            end = m.group(2)
            if cyl_n not in cyl_data:
                cyl_data[cyl_n] = {'he_pressure': None, 'ce_pressure': None, 'valves': []}
            if end == 'H':
                cyl_data[cyl_n]['he_pressure'] = col
            else:
                cyl_data[cyl_n]['ce_pressure'] = col
            continue

        # Specific valve column
        m = _VALVE_RE.match(seg)
        if m:
            cyl_n = int(m.group(1))
            end = m.group(2)
            valve_num = m.group(3)
            valve_type = m.group(4)  # S or D
            if cyl_n not in cyl_data:
                cyl_data[cyl_n] = {'he_pressure': None, 'ce_pressure': None, 'valves': []}
            end_label = "HE" if end == 'H' else "CE"
            type_label = "Suction" if valve_type == 'S' else "Discharge"
            valve_name = f"{end_label} {type_label} {valve_num} (US)"
            cyl_data[cyl_n]['valves'].append({'name': valve_name, 'curve': col})
            continue

        # General AE column (whole-cylinder ultrasonic, no specific valve)
        # These are not listed as valve_vibration_curves; skip them here.
        # (The general AE channel is still in the DataFrame and available for
        #  anomaly detection at the cylinder level.)
        m = _GENERAL_AE_RE.match(seg)
        if m:
            cyl_n = int(m.group(1))
            if cyl_n not in cyl_data:
                cyl_data[cyl_n] = {'he_pressure': None, 'ce_pressure': None, 'valves': []}
            # Intentionally not adding to valve_vibration_curves
            continue

    if not cyl_data:
        return None

    cylinders_config = []
    for cyl_n in sorted(cyl_data.keys()):
        d = cyl_data[cyl_n]
        he_pressure = d['he_pressure']
        ce_pressure = d['ce_pressure']
        pressure_curve = he_pressure or ce_pressure
        valve_curves = d['valves']

        if pressure_curve is None and not valve_curves:
            continue

        cylinders_config.append({
            "cylinder_name": f"Cylinder {cyl_n}",
            "pressure_curve": pressure_curve,
            "he_pressure_curve": he_pressure,
            "ce_pressure_curve": ce_pressure,
            "valve_vibration_curves": valve_curves,
            "bore": None,
            "rod_diameter": None,
            "stroke": None,
            "volume": None,
        })

    if not cylinders_config:
        return None

    return {
        "machine_id": machine_id,
        "model": "N/A",
        "serial_number": "N/A",
        "rated_rpm": "N/A",
        "rated_hp": "N/A",
        "num_cylinders": len(cylinders_config),
        "cylinders": cylinders_config,
    }
