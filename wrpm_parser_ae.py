"""
WRPM File Parser — AE Sensor Support for Leak Detection

Parses Windrock .wrpm files (ZIP archives) using the confirmed binary format spec.
Uses S&& as the TOC to seek directly into S$$ for per-sensor waveform extraction.

Machine Type Detection:
- Compressors (unit names ending in C, like 2C, 3C): 360° crank angle
- Engines (unit names ending in E, like 2E): 720° crank angle — excluded from AE leak detection

Format references: WRPM_FORMAT_SPEC.md (reverse-engineered from 3 production WRPM files)
"""

import zipfile
import struct
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from io import BytesIO


# ── Constants from confirmed spec ─────────────────────────────────────────────
WAVEFORM_HEADER_BYTES = 24      # per-block header before int16 samples
FULL_SCALE_G = 10.0             # AE ultrasonic full-scale (D6DYSETU.DAT line ~864)
FULL_SCALE_PSI = 2000.0         # Pressure full-scale default (D6CALFAC.DAT)
D6DYSEN_RECORD_BYTES = 44       # fixed sensor name record length
SAA_RECORD_BYTES = 38           # fixed S&& index record length (ASCII, no newlines)
MIN_WAVEFORM_SAMPLES = 10       # below this = scalar (temp/RPM), skip

# AE filter: name must contain "Ultra" and must NOT contain any of these
AE_EXCLUDE_WORDS = ('Power', 'Plug', 'Ignition', 'Ign')

# Column name suffix to match XML parser convention
AE_COLUMN_SUFFIX = 'ULTRASONIC G 36KHZ - 44KHZ'


def _is_ae_sensor(name: str) -> bool:
    """Return True if this sensor name is a compressor AE ultrasonic sensor."""
    return 'Ultra' in name and not any(w in name for w in AE_EXCLUDE_WORDS)


class WrpmParserAE:
    """
    WRPM parser using S&& TOC → S$$ waveform extraction.

    Public API (signatures preserved for compatibility with unified_data_loader.py):
        parse_to_dataframe() -> pd.DataFrame
        parse_pressure_to_dataframe() -> Optional[pd.DataFrame]
        get_curve_info() -> Dict
    """

    def __init__(self, wrpm_source):
        """
        Args:
            wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)
        """
        if isinstance(wrpm_source, (str, Path)):
            self.wrpm_path = Path(wrpm_source)
            self.file_obj = None
        else:
            self.file_obj = wrpm_source
            self.wrpm_path = None

        # Populated during parsing
        self.machine_id: Optional[str] = None
        self.date: Optional[datetime] = None
        self.session_name: Optional[str] = None
        self.full_scale_psi: float = FULL_SCALE_PSI
        self.full_scale_g: float = FULL_SCALE_G
        self.machine_type: Optional[str] = None   # 'compressor' or 'engine'
        self.crank_angle_range: int = 360
        self.is_engine: bool = False

    # ── ZIP access ─────────────────────────────────────────────────────────────

    def _get_zipfile(self) -> zipfile.ZipFile:
        if self.file_obj is not None:
            return zipfile.ZipFile(self.file_obj, 'r')
        return zipfile.ZipFile(self.wrpm_path, 'r')

    # ── Metadata parsers ───────────────────────────────────────────────────────

    def _parse_machine_id(self, z: zipfile.ZipFile) -> str:
        """Read D6NAME3.DAT and detect compressor vs engine."""
        try:
            if 'D6NAME3.DAT' not in z.namelist():
                self.machine_id = self.machine_id or "Unknown Machine"
                self._detect_machine_type()
                return self.machine_id

            raw = z.read('D6NAME3.DAT')
            text = raw.decode('ascii', errors='ignore').strip()
            # Newlines in the file act as a human-readable separator — flatten them
            self.machine_id = text.replace('\r\n', ' - ').replace('\n', ' - ').replace('\r', ' - ')
        except Exception:
            self.machine_id = self.machine_id or "Unknown Machine"

        self._detect_machine_type()
        return self.machine_id

    def _detect_machine_type(self):
        """Set machine_type, crank_angle_range, is_engine from machine_id."""
        mid = (self.machine_id or '').upper()

        engine_patterns = [r'UNIT\s*\d+\s*E\b', r'\b\d+\s*E\b', r'[-\s]E\b']
        for pat in engine_patterns:
            if re.search(pat, mid):
                self.machine_type = 'engine'
                self.crank_angle_range = 720
                self.is_engine = True
                return

        compressor_patterns = [r'UNIT\s*\d+\s*C\b', r'\b\d+\s*C\b', r'[-\s]C\b']
        for pat in compressor_patterns:
            if re.search(pat, mid):
                self.machine_type = 'compressor'
                self.crank_angle_range = 360
                self.is_engine = False
                return

        # Default to compressor
        self.machine_type = 'compressor'
        self.crank_angle_range = 360
        self.is_engine = False

    def _parse_date(self, z: zipfile.ZipFile) -> Optional[datetime]:
        """Extract collection date from session filename (S{YY}S{MMDD} pattern)."""
        pattern = r'S(\d{2})S(\d{2})(\d{2})'
        for filename in sorted(z.namelist()):
            m = re.search(pattern, filename)
            if m:
                try:
                    year = 2000 + int(m.group(1))
                    month = int(m.group(2))
                    day = int(m.group(3))
                    self.date = datetime(year, month, day)
                    return self.date
                except ValueError:
                    continue
        return None

    # ── Session selection ──────────────────────────────────────────────────────

    def _select_session(self, z: zipfile.ZipFile) -> str:
        """
        Return session base name (e.g. 'S25S0903').

        Prefers the latest session that contains compressor AE (Ultra) sensors.
        Falls back to the latest session overall if none have AE data.
        """
        saa_files = sorted(f for f in z.namelist() if f.endswith('.S&&'))
        if not saa_files:
            raise FileNotFoundError("No S&& index files found in WRPM archive")

        sensor_names = self._load_sensor_names(z)

        # Try sessions newest-first; pick the first one with AE Ultra sensors
        for saa_file in reversed(saa_files):
            candidate = saa_file.split('.')[0]
            sss_file = candidate + '.S$$'
            if sss_file not in z.namelist():
                continue
            try:
                raw_saa = z.read(saa_file)
                text = raw_saa.decode('ascii', errors='replace')
                n_recs = len(text) // SAA_RECORD_BYTES
                for i in range(1, n_recs):
                    rec = text[i * SAA_RECORD_BYTES:(i + 1) * SAA_RECORD_BYTES].split()
                    if len(rec) < 1:
                        continue
                    name = sensor_names.get(int(rec[0]), '')
                    if _is_ae_sensor(name):
                        self.session_name = candidate
                        return candidate
            except Exception:
                continue

        # Fallback: latest session
        session = saa_files[-1].split('.')[0]
        self.session_name = session
        return session

    # ── Sensor name table ──────────────────────────────────────────────────────

    def _load_sensor_names(self, z: zipfile.ZipFile) -> Dict[int, str]:
        """
        Build 1-based sensor index → name mapping.

        Primary source: D6DYSEN.DAT (44-byte fixed records, null-padded ASCII names).
        Fallback: D6_MACHINE.DAT (protobuf-encoded machine setup; sensor names are
                  length-delimited UTF-8 strings tagged with field 1 (0x0a) at the
                  start of each channel sub-message, stored sequentially so list
                  position == 1-based sensor index).

        If neither file is present, returns empty dict — callers use Sensor_{id} fallback.
        """
        files = z.namelist()
        sensor_names: Dict[int, str] = {}

        if 'D6DYSEN.DAT' in files:
            raw = z.read('D6DYSEN.DAT')
            n = len(raw) // D6DYSEN_RECORD_BYTES
            for i in range(n):
                chunk = raw[i * D6DYSEN_RECORD_BYTES: (i + 1) * D6DYSEN_RECORD_BYTES]
                name = chunk.split(b'\x00')[0].decode('ascii', errors='replace').strip()
                sensor_names[i + 1] = name  # 1-based
            return sensor_names

        if 'D6_MACHINE.DAT' in files:
            # Protobuf-encoded machine setup file (confirmed from Dwale Unit 3C).
            # Channel sub-messages are sequentially stored; each begins with a
            # field-1 length-delimited string (tag byte 0x0a + varint length + ASCII name).
            # Scan all such occurrences in file order; their position in the list
            # corresponds to 1-based sensor index.
            raw = z.read('D6_MACHINE.DAT')
            names_ordered: List[str] = []
            i = 0
            while i < len(raw) - 2:
                if raw[i] == 0x0A:
                    length = raw[i + 1]
                    # Plausible sensor name: 4–60 printable ASCII chars
                    if 4 <= length <= 60 and i + 2 + length <= len(raw):
                        name_bytes = raw[i + 2: i + 2 + length]
                        if all(32 <= b < 127 for b in name_bytes):
                            names_ordered.append(name_bytes.decode('ascii'))
                            i += 2 + length
                            continue
                i += 1

            for idx, name in enumerate(names_ordered, start=1):
                sensor_names[idx] = name

        return sensor_names

    # ── Core S&& / S$$ parser ─────────────────────────────────────────────────

    def _parse_ae_waveforms(self, z: zipfile.ZipFile, session: str) -> Dict[str, np.ndarray]:
        """
        Extract and average AE waveforms for all compressor Ultra sensors.

        Algorithm (spec Section 7):
          1. Parse S&& fixed-width ASCII records → (sensor_id, byte_offset) list
          2. Compute per-block deltas → n_samples = (delta - 24) // 2
          3. Filter to AE Ultra sensors, skip scalars (n_samples < MIN_WAVEFORM_SAMPLES)
          4. Average all collections per sensor (trim to min_len)

        Returns:
            Dict[sensor_label -> averaged float32 G array]
        """
        saa_file = session + '.S&&'
        sss_file = session + '.S$$'

        if sss_file not in z.namelist():
            raise FileNotFoundError(f"{sss_file} not found in WRPM archive")

        sensor_names = self._load_sensor_names(z)

        # Read both files into memory (S$$ can be ~2.4 MB for AE data)
        raw_saa = z.read(saa_file)
        raw_sss = z.read(sss_file)

        # Parse S&& fixed-width records (38 chars each, plain ASCII)
        text = raw_saa.decode('ascii', errors='replace')
        n_recs = len(text) // SAA_RECORD_BYTES
        all_recs = [text[i * SAA_RECORD_BYTES: (i + 1) * SAA_RECORD_BYTES].split()
                    for i in range(n_recs)]

        # Record 0 is a header ("1  1  1  1  1  1") — skip it
        data_recs = [r for r in all_recs[1:] if len(r) >= 3]

        if not data_recs:
            return {}

        # Extract byte offsets (field index 2 in each record)
        byte_offsets = [int(r[2]) for r in data_recs]

        # Compute block deltas: span from this offset to next; last block uses S$$ EOF
        n = len(byte_offsets)
        deltas = [byte_offsets[i + 1] - byte_offsets[i] for i in range(n - 1)]
        deltas.append(len(raw_sss) - byte_offsets[-1])

        # Group waveforms by sensor
        sensor_collections: Dict[str, List[np.ndarray]] = {}

        for rec, delta, byte_offset in zip(data_recs, deltas, byte_offsets):
            sensor_id = int(rec[0])

            # Skip blocks with invalid delta
            if delta < WAVEFORM_HEADER_BYTES:
                continue
            remainder = delta - WAVEFORM_HEADER_BYTES
            if remainder % 2 != 0:
                continue
            n_samples = remainder // 2

            # Skip scalar entries (temperature, RPM, etc.)
            if n_samples < MIN_WAVEFORM_SAMPLES:
                continue

            # Resolve sensor name — fallback for missing D6DYSEN.DAT (Dwale)
            name = sensor_names.get(sensor_id, f'Sensor_{sensor_id}')

            # AE Ultra compressor filter
            if 'Ultra' not in name:
                continue
            if any(word in name for word in AE_EXCLUDE_WORDS):
                continue

            # Extract raw int16 samples from S$$
            start = byte_offset + WAVEFORM_HEADER_BYTES
            end = start + n_samples * 2
            if end > len(raw_sss):
                continue  # Truncated file guard

            raw_block = raw_sss[start:end]
            samples = np.frombuffer(raw_block, dtype='<i2').astype(np.float32)

            # Convert to G units, remove DC offset, rectify to envelope
            # AE signals are AC — they oscillate around a DC baseline that varies
            # by sensor/electronics. The leak detector expects positive envelope
            # amplitude (same format as XML Curves files), not raw signed samples.
            g_samples = samples / 32768.0 * FULL_SCALE_G
            g_samples -= g_samples.mean()   # remove DC offset
            g_samples = np.abs(g_samples)   # rectify → envelope

            label = f'{sensor_id}:{name}'
            if label not in sensor_collections:
                sensor_collections[label] = []
            sensor_collections[label].append(g_samples)

        # Average all collections per sensor (trim to min_len to handle RPM drift)
        result: Dict[str, np.ndarray] = {}
        for label, waveforms in sensor_collections.items():
            if not waveforms:
                continue
            min_len = min(len(w) for w in waveforms)
            stacked = np.stack([w[:min_len] for w in waveforms], axis=0)
            result[label] = stacked.mean(axis=0)

        return result

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        Parse WRPM file and return AE waveform DataFrame compatible with XML parser format.

        Returns:
            DataFrame with columns:
              - 'Crank Angle': 0–360 float values (compressor) or 0–720 (engine)
              - Per-sensor columns named:
                '{machine_id} - {sensor_name}.ULTRASONIC G 36KHZ - 44KHZ.{sensor_idx}'
              Values are in G units (float32), averaged over all collections.
        """
        with self._get_zipfile() as z:
            self._parse_machine_id(z)
            self._parse_date(z)
            session = self._select_session(z)

            ae_waveforms = self._parse_ae_waveforms(z, session)

        if not ae_waveforms:
            # Return empty DataFrame with just Crank Angle — no AE sensors in this session
            n_pts = self.crank_angle_range + 1  # e.g. 361 for a 360° compressor
            angles = np.linspace(0, self.crank_angle_range, n_pts, endpoint=False)
            return pd.DataFrame({'Crank Angle': angles})

        # Determine output length from waveforms (use minimum across all sensors)
        min_len = min(len(w) for w in ae_waveforms.values())
        angles = np.linspace(0, self.crank_angle_range, min_len, endpoint=False)

        machine_id = self.machine_id or "Unknown"
        df_data: Dict = {'Crank Angle': angles}

        for label, waveform in ae_waveforms.items():
            # label = '{sensor_idx}:{sensor_name}'
            sensor_idx, sensor_name = label.split(':', 1)
            col_name = f'{machine_id} - {sensor_name}.{AE_COLUMN_SUFFIX}.{sensor_idx}'
            df_data[col_name] = waveform[:min_len].astype(np.float32)

        return pd.DataFrame(df_data)

    def parse_pressure_to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Parse WRPM file and extract PVPT pressure waveforms from S$$ via the P&& index.

        Uses the same S&&/S$$ architecture as AE parsing but reads from P&& / P$$
        session files (same binary format, different sensor types).

        Returns:
            DataFrame with 'Crank Angle' + pressure curve columns, or None if unavailable.
        """
        with self._get_zipfile() as z:
            self._parse_machine_id(z)
            self._parse_date(z)
            session = self._select_session(z)

            files = z.namelist()
            paa_file = session + '.P&&'
            pss_file = session + '.P$$'

            # Fall back to S$$ / S&& if P files don't exist
            if paa_file not in files or pss_file not in files:
                # Try legacy approach: look for S$$ and treat non-Ultra sensors as pressure
                sss_file = session + '.S$$'
                if sss_file not in files:
                    return None
                return self._parse_pressure_from_sss(z, session)

            sensor_names = self._load_sensor_names(z)

            raw_paa = z.read(paa_file)
            raw_pss = z.read(pss_file)

            text = raw_paa.decode('ascii', errors='replace')
            n_recs = len(text) // SAA_RECORD_BYTES
            all_recs = [text[i * SAA_RECORD_BYTES: (i + 1) * SAA_RECORD_BYTES].split()
                        for i in range(n_recs)]
            data_recs = [r for r in all_recs[1:] if len(r) >= 3]

            if not data_recs:
                return None

            byte_offsets = [int(r[2]) for r in data_recs]
            n = len(byte_offsets)
            deltas = [byte_offsets[i + 1] - byte_offsets[i] for i in range(n - 1)]
            deltas.append(len(raw_pss) - byte_offsets[-1])

            sensor_collections: Dict[str, List[np.ndarray]] = {}

            for rec, delta, byte_offset in zip(data_recs, deltas, byte_offsets):
                sensor_id = int(rec[0])
                if delta < WAVEFORM_HEADER_BYTES:
                    continue
                remainder = delta - WAVEFORM_HEADER_BYTES
                if remainder % 2 != 0:
                    continue
                n_samples = remainder // 2
                if n_samples < MIN_WAVEFORM_SAMPLES:
                    continue

                name = sensor_names.get(sensor_id, f'Sensor_{sensor_id}')

                # Keep only COMPRESSOR pressure sensors (must have both 'Comp' and 'Pressure')
                # Exclude engine power cylinder pressure sensors ('Power * Pressure')
                if not ('Comp' in name and 'Pressure' in name):
                    continue

                start = byte_offset + WAVEFORM_HEADER_BYTES
                end = start + n_samples * 2
                if end > len(raw_pss):
                    continue

                raw_block = raw_pss[start:end]
                samples = np.frombuffer(raw_block, dtype='<i2').astype(np.float32)
                psi_samples = samples / 32768.0 * self.full_scale_psi
                psi_samples -= psi_samples.mean()  # remove DC offset (PVPT sensor, no static reference)
                # Clip wrap-around spike artifact at cycle boundary (~359°)
                p99 = float(np.percentile(np.abs(psi_samples), 99))
                clip_limit = p99 * 1.5
                if clip_limit > 0:
                    psi_samples = np.clip(psi_samples, -clip_limit, clip_limit)

                label = f'{sensor_id}:{name}'
                if label not in sensor_collections:
                    sensor_collections[label] = []
                sensor_collections[label].append(psi_samples)

            if not sensor_collections:
                # P$$ exists but contains no compressor pressure sensors (e.g. engine-only P$$)
                # Fall back to S$$ which may contain compressor pressure channels
                return self._parse_pressure_from_sss(z, session)

            averaged: Dict[str, np.ndarray] = {}
            for label, waves in sensor_collections.items():
                min_len = min(len(w) for w in waves)
                stacked = np.stack([w[:min_len] for w in waves], axis=0)
                averaged[label] = stacked.mean(axis=0)

            min_len = min(len(w) for w in averaged.values())
            angles = np.linspace(0, self.crank_angle_range, min_len, endpoint=False)
            machine_id = self.machine_id or "Unknown"
            df_data: Dict = {'Crank Angle': angles}

            for label, waveform in averaged.items():
                sensor_idx, sensor_name = label.split(':', 1)
                col_name = f'{machine_id} - {sensor_name}.PVPT.{sensor_idx}'
                df_data[col_name] = waveform[:min_len].astype(np.float32)

            return pd.DataFrame(df_data)

    def _parse_pressure_from_sss(self, z: zipfile.ZipFile, session: str) -> Optional[pd.DataFrame]:
        """
        Fallback: extract pressure waveforms from S$$ when P$$ is absent.
        Looks for sensors containing 'Pressure' in D6DYSEN.DAT names.
        """
        saa_file = session + '.S&&'
        sss_file = session + '.S$$'

        if sss_file not in z.namelist():
            return None

        sensor_names = self._load_sensor_names(z)
        raw_saa = z.read(saa_file)
        raw_sss = z.read(sss_file)

        text = raw_saa.decode('ascii', errors='replace')
        n_recs = len(text) // SAA_RECORD_BYTES
        all_recs = [text[i * SAA_RECORD_BYTES: (i + 1) * SAA_RECORD_BYTES].split()
                    for i in range(n_recs)]
        data_recs = [r for r in all_recs[1:] if len(r) >= 3]

        if not data_recs:
            return None

        byte_offsets = [int(r[2]) for r in data_recs]
        n = len(byte_offsets)
        deltas = [byte_offsets[i + 1] - byte_offsets[i] for i in range(n - 1)]
        deltas.append(len(raw_sss) - byte_offsets[-1])

        sensor_collections: Dict[str, List[np.ndarray]] = {}

        for rec, delta, byte_offset in zip(data_recs, deltas, byte_offsets):
            sensor_id = int(rec[0])
            if delta < WAVEFORM_HEADER_BYTES:
                continue
            remainder = delta - WAVEFORM_HEADER_BYTES
            if remainder % 2 != 0:
                continue
            n_samples = remainder // 2
            if n_samples < MIN_WAVEFORM_SAMPLES:
                continue

            name = sensor_names.get(sensor_id, f'Sensor_{sensor_id}')
            # Only compressor pressure sensors (Comp N H/C Pressure); skip engine power sensors
            if not ('Comp' in name and 'Pressure' in name):
                continue

            start = byte_offset + WAVEFORM_HEADER_BYTES
            end = start + n_samples * 2
            if end > len(raw_sss):
                continue

            raw_block = raw_sss[start:end]
            samples = np.frombuffer(raw_block, dtype='<i2').astype(np.float32)
            psi_samples = samples / 32768.0 * self.full_scale_psi
            psi_samples -= psi_samples.mean()  # remove DC offset (PVPT sensor, no static reference)
            # Clip wrap-around spike artifact at cycle boundary (~359°)
            p99 = float(np.percentile(np.abs(psi_samples), 99))
            clip_limit = p99 * 1.5
            if clip_limit > 0:
                psi_samples = np.clip(psi_samples, -clip_limit, clip_limit)

            label = f'{sensor_id}:{name}'
            if label not in sensor_collections:
                sensor_collections[label] = []
            sensor_collections[label].append(psi_samples)

        if not sensor_collections:
            return None

        averaged: Dict[str, np.ndarray] = {}
        for label, waves in sensor_collections.items():
            min_len = min(len(w) for w in waves)
            stacked = np.stack([w[:min_len] for w in waves], axis=0)
            averaged[label] = stacked.mean(axis=0)

        min_len = min(len(w) for w in averaged.values())
        angles = np.linspace(0, self.crank_angle_range, min_len, endpoint=False)
        machine_id = self.machine_id or "Unknown"
        df_data: Dict = {'Crank Angle': angles}

        for label, waveform in averaged.items():
            sensor_idx, sensor_name = label.split(':', 1)
            col_name = f'{machine_id} - {sensor_name}.PVPT.{sensor_idx}'
            df_data[col_name] = waveform[:min_len].astype(np.float32)

        return pd.DataFrame(df_data)

    def get_curve_info(self) -> Dict:
        """
        Return metadata about this WRPM file.

        Returns:
            Dict with keys: total_curves, ae_curves, data_points, crank_angle_range,
            machine_id, date, file_type, machine_type, is_engine, has_pressure_data,
            session_name
        """
        with self._get_zipfile() as z:
            self._parse_machine_id(z)
            self._parse_date(z)
            session = self._select_session(z)

            files = z.namelist()
            has_pressure = (session + '.P$$') in files or (session + '.S$$') in files

            try:
                ae_waveforms = self._parse_ae_waveforms(z, session)
                ae_sensor_labels = list(ae_waveforms.keys())
                n_pts = min((len(w) for w in ae_waveforms.values()), default=0)
            except Exception:
                ae_sensor_labels = []
                n_pts = 0

        angle_range = f'0-{self.crank_angle_range}°'

        return {
            'total_curves': len(ae_sensor_labels),
            'ae_curves': ae_sensor_labels,
            'data_points': n_pts,
            'crank_angle_range': angle_range,
            'machine_id': self.machine_id,
            'date': self.date,
            'file_type': 'WRPM',
            'machine_type': self.machine_type,
            'is_engine': self.is_engine,
            'has_pressure_data': has_pressure,
            'session_name': self.session_name,
        }


# ── Module-level convenience functions (public API) ────────────────────────────

def parse_wrpm_to_dataframe(wrpm_source) -> pd.DataFrame:
    """
    Parse a WRPM file and return a DataFrame of AE sensor waveforms.

    Args:
        wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)

    Returns:
        DataFrame with 'Crank Angle' column + one column per AE sensor in G units
    """
    return WrpmParserAE(wrpm_source).parse_to_dataframe()


def get_wrpm_curve_info(wrpm_source) -> Dict:
    """
    Return metadata about a WRPM file without parsing all waveform data.

    Args:
        wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)

    Returns:
        Dict with machine_id, date, is_engine, ae_curves, etc.
    """
    return WrpmParserAE(wrpm_source).get_curve_info()


def parse_wrpm_pressure_to_dataframe(wrpm_source) -> Optional[pd.DataFrame]:
    """
    Parse WRPM file and extract PVPT pressure waveforms.

    Args:
        wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)

    Returns:
        DataFrame with 'Crank Angle' + pressure columns in PSI, or None if unavailable
    """
    return WrpmParserAE(wrpm_source).parse_pressure_to_dataframe()


def is_engine_file(wrpm_source) -> bool:
    """
    Return True if the WRPM file is from an engine unit (vs compressor).

    Engine files are excluded from AE valve leak detection (different crank cycle).

    Args:
        wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)
    """
    return WrpmParserAE(wrpm_source).get_curve_info().get('is_engine', False)
