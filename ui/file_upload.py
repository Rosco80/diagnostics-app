"""
File upload section: validation, preview, and styled UI.

Supports two upload modes:
  - XML (3-file set: Curves, Levels, Source) — original Windrock export format
  - WRPM (single .wrpm ZIP archive) — newer Windrock format with binary AE waveforms
"""

import xml.etree.ElementTree as ET
import streamlit as st

from core.xml_parser import (
    find_xml_value, load_all_curves_data, extract_rpm_from_source,
    load_wrpm_curves_data,
)
from core.config_discovery import auto_discover_configuration
from ui.components import render_section_header


def validate_xml_files(uploaded_files):
    """
    Validates uploaded XML files and returns validation results - ROBUST VERSION.
    """
    validation_results = {
        'is_valid': False,
        'files_found': {},
        'missing_files': [],
        'file_info': {},
        'errors': []
    }

    if len(uploaded_files) != 3:
        validation_results['errors'].append(f"Expected 3 files, got {len(uploaded_files)}")
        return validation_results

    # Check for required file types
    required_files = ['curves', 'levels', 'source']
    found_files = {}

    for file in uploaded_files:
        filename_lower = file.name.lower()
        if 'curves' in filename_lower:
            found_files['curves'] = file
        elif 'levels' in filename_lower:
            found_files['levels'] = file
        elif 'source' in filename_lower:
            found_files['source'] = file

    # Check what's missing
    missing = [file_type for file_type in required_files if file_type not in found_files]
    validation_results['missing_files'] = missing
    validation_results['files_found'] = found_files

    if missing:
        validation_results['errors'].append(f"Missing required files: {', '.join(missing)}")
        return validation_results

    # Validate each XML file with robust error handling
    for file_type, file in found_files.items():
        try:
            content = file.getvalue().decode('utf-8')

            # Basic XML validation
            root = ET.fromstring(content)

            # File-specific validation with safe fallbacks
            if file_type == 'curves':
                try:
                    # Count data elements safely
                    data_elements = root.findall('.//Data')
                    curve_count = len([elem for elem in data_elements if elem.text and elem.text.strip()])
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'data_points': curve_count,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'data_points': 0,
                        'status': 'Valid'
                    }

            elif file_type == 'levels':
                try:
                    # Extract machine info safely
                    machine_info = None
                    try:
                        machine_info = find_xml_value(root, 'Levels', 'Machine', 2)
                    except Exception:
                        pass

                    if not machine_info or machine_info == 'N/A':
                        machine_info = 'Unknown'

                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'machine_id': machine_info,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'machine_id': 'Unknown',
                        'status': 'Valid'
                    }

            elif file_type == 'source':
                try:
                    # Count configuration entries safely
                    config_count = 0
                    try:
                        # Safe iteration through elements
                        for elem in root.iter():
                            if hasattr(elem, 'text') and elem.text and 'CYLINDER' in str(elem.text):
                                config_count += 1
                    except Exception:
                        config_count = 0

                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'config_entries': config_count,
                        'status': 'Valid'
                    }
                except Exception:
                    validation_results['file_info'][file_type] = {
                        'size_kb': len(content) / 1024,
                        'config_entries': 0,
                        'status': 'Valid'
                    }

        except ET.ParseError:
            validation_results['errors'].append(f"{file_type.title()} file: Invalid XML format")
            validation_results['file_info'][file_type] = {'status': 'Invalid XML', 'error': 'XML parsing failed'}
        except UnicodeDecodeError:
            validation_results['errors'].append(f"{file_type.title()} file: Invalid file encoding")
            validation_results['file_info'][file_type] = {'status': 'Encoding Error', 'error': 'Cannot decode file'}
        except Exception:
            validation_results['errors'].append(f"{file_type.title()} file: Unexpected error")
            validation_results['file_info'][file_type] = {'status': 'Error', 'error': 'Processing failed'}

    # Set overall validation status
    validation_results['is_valid'] = len(validation_results['errors']) == 0

    return validation_results


def extract_preview_info(files_content):
    """
    Extracts key information for preview display - FIXED: RPM now pulled from source.xml.
    """
    preview_info = {
        'machine_id': 'Unknown',
        'rpm': 'Unknown',
        'cylinder_count': 0,
        'total_curves': 0,
        'file_sizes': {},
        'date_time': 'Unknown'
    }

    # LEVELS FILE - Get date/time only
    if 'levels' in files_content:
        try:
            levels_root = ET.fromstring(files_content['levels'])
            for elem in levels_root.iter():
                if hasattr(elem, 'text') and elem.text and '/' in str(elem.text) and len(str(elem.text)) > 8:
                    preview_info['date_time'] = str(elem.text)
                    break
        except Exception:
            pass
        preview_info['file_sizes']['levels'] = len(files_content['levels']) / 1024

    # CURVES FILE - Count curves
    if 'curves' in files_content:
        try:
            curves_root = ET.fromstring(files_content['curves'])

            # Count curves
            curve_count = 0
            for i, elem in enumerate(curves_root.iter()):
                if hasattr(elem, 'text') and elem.text:
                    text_upper = str(elem.text).upper()
                    if any(k in text_upper for k in ['PRESSURE', 'VIBRATION', 'PHASED']):
                        curve_count += 1
                if i > 200:
                    break
            preview_info['total_curves'] = curve_count

        except Exception:
            pass
        preview_info['file_sizes']['curves'] = len(files_content['curves']) / 1024

    # SOURCE FILE - Machine ID, RPM, Cylinder Count
    if 'source' in files_content:
        try:
            source_root = ET.fromstring(files_content['source'])

            # Extract RPM from source
            rpm = extract_rpm_from_source(files_content['source'])
            if rpm and rpm not in ['N/A', '', 'Unknown']:
                preview_info['rpm'] = rpm

            # Try extracting machine ID directly
            for elem in source_root.iter():
                if hasattr(elem, 'text') and elem.text:
                    text = str(elem.text).strip()
                    if ('-' in text and len(text) < 20 and len(text) > 3 and
                            any(c.isalnum() for c in text) and
                            not any(w in text.upper() for w in ['CYLINDER', 'PRESSURE', 'TEMPERATURE', 'VALVE', 'COMPRESSOR'])):
                        preview_info['machine_id'] = text
                        break

            # Auto-discovery for cylinder count
            try:
                curves_content = files_content.get('curves', '')
                if curves_content:
                    df, curve_names = load_all_curves_data(curves_content)
                    if df is not None and curve_names:
                        config = auto_discover_configuration(files_content['source'], curve_names)
                        if config and 'cylinders' in config:
                            preview_info['cylinder_count'] = len(config['cylinders'])
                            if preview_info['machine_id'] == 'Unknown' and config.get('machine_id'):
                                preview_info['machine_id'] = config['machine_id']
            except Exception:
                # Fallback: simple bore count
                bore_count = 0
                for elem in source_root.iter():
                    if hasattr(elem, 'text') and elem.text and 'BORE' in elem.text.upper():
                        bore_count += 1
                preview_info['cylinder_count'] = min(bore_count, 10)

        except Exception:
            pass
        preview_info['file_sizes']['source'] = len(files_content['source']) / 1024

    return preview_info


def enhanced_file_upload_section():
    """
    Enhanced file upload with validation, preview, and styled UI.
    """
    render_section_header("📁 1. Data Upload")

    # Check if we already have validated files in session state
    if 'validated_files' in st.session_state and st.session_state.validated_files:
        files_content = st.session_state.validated_files

        # ── WRPM already-loaded display ────────────────────────────────────────
        if files_content.get('_wrpm'):
            curves_dict = files_content['curves_dict']
            machine_id = files_content['machine_id']
            ae_cols = [k for k in curves_dict if k != 'Crank Angle']
            st.success(
                f"WRPM loaded — Machine: **{machine_id}** | AE sensors: **{len(ae_cols)}**"
            )
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    if st.button("Upload New Files", use_container_width=True):
                        st.session_state.file_uploader_key += 1
                        st.session_state.active_session_id = None
                        if 'validated_files' in st.session_state:
                            del st.session_state.validated_files
                        if 'wrpm_curves_data' in st.session_state:
                            del st.session_state.wrpm_curves_data
                        if 'auto_discover_config' in st.session_state:
                            del st.session_state['auto_discover_config']
                        if 'analysis_results' in st.session_state:
                            st.session_state.analysis_results = None
                        st.cache_data.clear()
                        st.rerun()
            return files_content

        # ── XML already-loaded display ─────────────────────────────────────────
        st.success("Files already loaded and validated!")

        preview_info = extract_preview_info(files_content)

        st.markdown("""
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; font-size: 0.9rem;">
    <div style="flex: 1 1 45%;">
        <strong>Machine ID:</strong><br>{machine_id}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Data Curves:</strong><br>{curves}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Cylinders:</strong><br>{cyl}
    </div>
    <div style="flex: 1 1 45%;">
        <strong>Total Size:</strong><br>{size} KB
    </div>
</div>
""".format(
            machine_id=preview_info['machine_id'],
            curves=preview_info['total_curves'],
            cyl=preview_info['cylinder_count'],
            size=f"{sum(preview_info['file_sizes'].values()):.1f}"
        ), unsafe_allow_html=True)

        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if st.button("Upload New Files", use_container_width=True):
                    st.session_state.file_uploader_key += 1
                    st.session_state.active_session_id = None
                    if 'validated_files' in st.session_state:
                        del st.session_state.validated_files
                    if 'wrpm_curves_data' in st.session_state:
                        del st.session_state.wrpm_curves_data
                    if 'auto_discover_config' in st.session_state:
                        del st.session_state['auto_discover_config']
                    if 'analysis_results' in st.session_state:
                        st.session_state.analysis_results = None
                    st.cache_data.clear()  # Clear Streamlit cache to load fresh data
                    st.rerun()

        return files_content

    # Upload input — accepts both legacy 3-file XML sets and single WRPM archives
    uploaded_files = st.file_uploader(
        "Upload Curves, Levels, Source XML files  —  or a single .wrpm file",
        type=["xml", "wrpm"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        help=(
            "XML mode: upload exactly 3 files (Curves.xml, Levels.xml, Source.xml).\n"
            "WRPM mode: upload a single .wrpm archive file."
        )
    )

    if st.button("Start New Analysis / Clear Files"):
        st.session_state.file_uploader_key += 1
        st.session_state.active_session_id = None
        if 'validated_files' in st.session_state:
            del st.session_state.validated_files
        if 'wrpm_curves_data' in st.session_state:
            del st.session_state.wrpm_curves_data
        if 'analysis_results' in st.session_state:
            st.session_state.analysis_results = None
        if 'auto_discover_config' in st.session_state:
            del st.session_state['auto_discover_config']
        st.cache_data.clear()  # Clear Streamlit cache to load fresh data
        st.rerun()

    # ── WRPM branch ────────────────────────────────────────────────────────────
    if uploaded_files and len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith('.wrpm'):
        wrpm_file = uploaded_files[0]
        st.info(f"WRPM file detected: {wrpm_file.name}")

        with st.spinner("Parsing WRPM file..."):
            try:
                wrpm_bytes = wrpm_file.getvalue()
                curves_dict, machine_id, is_engine = load_wrpm_curves_data(wrpm_bytes)
            except Exception as exc:
                st.error(f"Failed to parse WRPM file: {exc}")
                return None

        if is_engine:
            st.warning(
                f"Engine file detected (machine: {machine_id}). "
                "Engine units run a 720-degree crank cycle and are not supported "
                "by the compressor valve leak detector. Please upload a compressor WRPM file."
            )
            return None

        # Count AE sensor columns (everything except Crank Angle)
        ae_sensor_cols = [k for k in curves_dict if k != 'Crank Angle']
        n_sensors = len(ae_sensor_cols)
        n_points = len(curves_dict.get('Crank Angle', []))

        st.success(
            f"WRPM loaded — Machine: **{machine_id}** | "
            f"AE sensors: **{n_sensors}** | "
            f"Crank angle points: **{n_points}**"
        )

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if st.button("Analyse WRPM Data", type="primary", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.active_session_id = None
                if 'auto_discover_config' in st.session_state:
                    del st.session_state['auto_discover_config']
                st.session_state.wrpm_curves_data = {
                    'curves_dict': curves_dict,
                    'machine_id': machine_id,
                    'is_engine': is_engine,
                }
                # Store in validated_files format so app.py routing works
                st.session_state.validated_files = {
                    '_wrpm': True,
                    'curves_dict': curves_dict,
                    'machine_id': machine_id,
                    'is_engine': is_engine,
                }
                return st.session_state.validated_files

        return None

    if uploaded_files:
        if len(uploaded_files) != 3:
            st.error(f"❌ Please upload exactly 3 XML files. You uploaded {len(uploaded_files)} files.")
            st.info("💡 Required files: Curves.xml, Levels.xml, Source.xml")
            return None

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔍 Validating uploaded files...")
        progress_bar.progress(25)
        validation_results = validate_xml_files(uploaded_files)

        if not validation_results['is_valid']:
            progress_bar.progress(100)
            st.error("❌ File validation failed!")
            for error in validation_results['errors']:
                st.error(f"• {error}")
            if validation_results['files_found']:
                st.info("✅ Files detected:")
                for file_type in validation_results['files_found']:
                    st.success(f"• {file_type.title()} file found")
            return None

        # Extract and Preview
        status_text.text("📄 Processing file contents...")
        progress_bar.progress(50)

        files_content = {}
        for file in uploaded_files:
            name = file.name.lower()
            if 'curves' in name:
                files_content['curves'] = file.getvalue().decode('utf-8')
            elif 'levels' in name:
                files_content['levels'] = file.getvalue().decode('utf-8')
            elif 'source' in name:
                files_content['source'] = file.getvalue().decode('utf-8')

        status_text.text("🔍 Generating preview...")
        progress_bar.progress(75)
        preview_info = extract_preview_info(files_content)

        progress_bar.progress(100)
        status_text.text("✅ Files ready for analysis!")
        st.success("✅ All files validated successfully!")

        st.markdown("### 📋 Data Preview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Machine Information:**
            - **ID:** {preview_info['machine_id']}
            - **RPM:** {preview_info['rpm']}
            - **Date:** {preview_info['date_time']}
            """)

        with col2:
            st.markdown(f"""
            **Data Summary:**
            - **Cylinders:** {preview_info['cylinder_count']}
            - **Data Curves:** {preview_info['total_curves']}
            - **Total Size:** {sum(preview_info['file_sizes'].values()):.1f} KB
            """)

        # File status
        st.markdown("**File Details:**")
        file_details = []
        for file_type, size_kb in preview_info['file_sizes'].items():
            status = "✅" if size_kb > 0 else "⚠️"
            file_details.append(f"{status} **{file_type.title()}:** {size_kb:.1f} KB")
        st.markdown(" | ".join(file_details))

        # Warnings
        warnings = []
        if preview_info['machine_id'] == 'Unknown':
            warnings.append("⚠️ Machine ID not detected")
        if preview_info['cylinder_count'] == 0:
            warnings.append("⚠️ No cylinders detected")
        if preview_info['total_curves'] == 0:
            warnings.append("⚠️ No data curves detected")

        if warnings:
            st.warning("**Data Quality Warnings:**\n" + "\n".join(warnings))

        # Expander for file info
        with st.expander("🔍 Detailed Technical Information"):
            for file_type, info in validation_results['file_info'].items():
                st.markdown(f"**{file_type.title()} File Analysis:**")
                if info['status'] == 'Valid':
                    st.write("• Status: ✅ Valid XML structure")
                    st.write(f"• File size: {info['size_kb']:.1f} KB")
                    if file_type == 'curves':
                        st.write(f"• Data elements: {info['data_points']}")
                        st.write(f"• Detected curves: {preview_info['total_curves']}")
                    elif file_type == 'levels':
                        st.write(f"• Machine ID: {preview_info['machine_id']}")
                        st.write(f"• Recording date: {preview_info['date_time']}")
                    elif file_type == 'source':
                        st.write(f"• Configuration entries: {info['config_entries']}")
                        st.write(f"• Detected cylinders: {preview_info['cylinder_count']}")
                else:
                    st.error(f"• Status: ❌ {info['status']}")
                    if 'error' in info:
                        st.error(f"• Error: {info['error']}")
                st.markdown("---")

        st.markdown("---")
        if warnings:
            st.warning("⚠️ **You can proceed, but please check the warnings above**")
        else:
            st.success("✅ **Data looks good! Ready to analyze**")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if st.button("🚀 Analyse", type="primary", use_container_width=True):
                # Clear old analysis results when new files are uploaded
                st.session_state.analysis_results = None
                st.session_state.active_session_id = None
                if 'auto_discover_config' in st.session_state:
                    del st.session_state['auto_discover_config']
                st.session_state.validated_files = files_content
                st.markdown('</div>', unsafe_allow_html=True)
                return files_content

        st.markdown('</div>', unsafe_allow_html=True)
        return None

    else:
        st.info("👆 Please upload your 3 XML files to begin")
        st.markdown('</div>', unsafe_allow_html=True)
        return None
