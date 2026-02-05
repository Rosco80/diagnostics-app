# Infrastructure Audit Report
## Week 1: System Audit & Data Preparation

**Date:** November 11, 2025
**Project:** AI-Powered Valve Leak Detection - 4-Week Pilot
**Auditor:** Andrea

---

## Executive Summary

This audit reviews the existing codebase (`my-project`) to identify reusable components for the leak detection pilot. The parent project contains a fully functional machine diagnostics system with XML parsing, database integration, and Streamlit dashboard (~11,700 lines across 3 app versions). **Key Finding:** Approximately 60-70% of the infrastructure can be reused or adapted, significantly reducing Week 2-4 development time.

### Status Overview
- ✅ **Fully Reusable:** XML parser, database client, visualization framework
- ⚠️ **Needs Adaptation:** Dashboard UI, database schema
- ❌ **Build from Scratch:** Feature extraction, binary leak classifier, inference pipeline
- 🔮 **Future Phase:** WRPM parser (Phase 2, pending client's AE data availability)

---

## 1. Code Repositories Reviewed

### Main Application Files (Parent `my-project` folder)

1. **latestMVP.py** (3,767 lines)
   - Most recent stable version
   - Full Streamlit dashboard implementation
   - Turso database integration
   - XML file processing
   - Isolation Forest anomaly detection
   - Report generation (PDF)

2. **currentapp.py** (3,985 lines)
   - Alternative version with additional features
   - Similar structure to latestMVP.py

3. **app.py** (3,983 lines)
   - Original implementation
   - Legacy codebase

### Modular Components

4. **phase1-modular/app/core/xml_parser.py** (486 lines)
   - **Status:** ✅ Excellent, fully reusable
   - Clean, well-documented XML parsing module
   - Functions:
     - `validate_xml_files()` - Validates uploaded Curves, Levels, Source files
     - `load_curves_data()` - Extracts waveform data into Pandas DataFrame
     - `extract_rpm()` - Extracts RPM metadata
     - `auto_discover_configuration()` - Discovers machine config (cylinders, valves, sensors)
     - `find_xml_value()` - Generic XML value extraction utility
   - **Recommendation:** Copy directly to `Leak_detector/app/core/xml_parser.py` with minimal changes

5. **app/core/wrpm_parser.py** (~200+ lines)
   - **Status:** 🔮 Available for Phase 2
   - Parses Windrock .wrpm files (ZIP archives)
   - Extracts pressure waveforms, calibration, metadata
   - **Limitation:** Current client WRPM samples lack AE data (pressure only)
   - **Recommendation:** Defer to Phase 2 if client provides WRPM files with AE sensors

---

## 2. Reusable Components

### ✅ Category A: Ready to Use (Copy & Go)

#### 2.1 XML Parser Module
**File:** `phase1-modular/app/core/xml_parser.py`
**Reusability:** 100%

**Key Functions:**
```python
# File validation
validation_results = validate_xml_files(uploaded_files)
# Returns: {'is_valid': True/False, 'files_found': {...}, 'missing_files': [...], 'errors': [...]}

# Load waveform data
curves_df, column_names = load_curves_data(curves_xml_content)
# Returns: DataFrame with 'Crank Angle' + all sensor columns

# Extract metadata
rpm = extract_rpm(levels_xml_content)
config = auto_discover_configuration(source_xml_content, column_names)
# config contains: machine_id, cylinders, valve locations, sensor mappings
```

**Why This Matters:**
- Leak detection requires AE waveform extraction → `load_curves_data()` provides this
- Inference pipeline needs to identify which columns are AE sensors → `auto_discover_configuration()` provides sensor mapping
- Dashboard needs file validation → `validate_xml_files()` handles this

**Action Items:**
- [x] Copy to `Leak_detector/app/core/xml_parser.py`
- [ ] Add docstring updates referencing leak detection use case
- [ ] Write unit tests specific to leak detection

#### 2.2 Database Integration
**File:** `latestMVP.py` lines 64-88
**Reusability:** 80% (needs schema extensions)

**Current Database Schema (Turso/libSQL):**
```sql
-- Existing tables (fully reusable)
sessions          -- Store analysis sessions (timestamp, machine_id, rpm)
waveform_data     -- Store raw waveform samples
configs           -- Store model configuration (thresholds, contamination factor)
alerts            -- Store critical alerts

-- Tables needing modification
analyses          -- ADD: leak_probability, model_version
labels            -- RENAME/EXTEND: Add structured fault tags

-- New tables needed for leak detection
model_metadata    -- Store model version, accuracy, training date, feature names
predictions       -- Store leak predictions (session_id, valve_id, probability, status, confidence)
training_samples  -- Store labeled training data for model retraining
```

**Action Items:**
- [ ] Copy `init_db()` function to `Leak_detector/app/core/database.py`
- [ ] Add schema migration for leak detection tables
- [ ] Test database connectivity with Turso credentials

#### 2.3 Visualization Framework
**File:** `latestMVP.py` (Plotly integration)
**Reusability:** 95%

**Existing Capabilities:**
- Plotly waveform charts (line plots, scatter)
- Multi-subplot layouts for comparing cylinders/valves
- Interactive zoom, pan, hover tooltips
- Custom CSS styling (`style.css`)

**What Can Be Reused:**
```python
# Waveform plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=crank_angles, y=waveform, mode='lines', name='AE Signal'))
fig.update_layout(title='Valve Leak Detection', xaxis_title='Crank Angle (deg)', yaxis_title='Amplitude (G)')

# Multi-valve comparison
fig = make_subplots(rows=2, cols=2, subplot_titles=valve_names)
for i, valve in enumerate(valves):
    fig.add_trace(go.Scatter(...), row=row, col=col)
```

**Action Items:**
- [ ] Extract reusable plotting functions to `Leak_detector/app/dashboard/plotting.py`
- [ ] Copy `style.css` to `Leak_detector/`

#### 2.4 Streamlit Dashboard Structure
**File:** `latestMVP.py`
**Reusability:** 80% (structure reusable, content needs updating)

**Reusable UI Patterns:**
- File upload section with validation
- Session state management
- Sidebar configuration
- Multi-column layouts
- Progress indicators
- Download buttons (CSV/PDF export)

**What Needs Changing:**
- Replace Isolation Forest anomaly detection with leak detection model inference
- Add leak probability gauges/indicators
- Color-code valve status (green/yellow/red)
- Update dashboard title/branding

**Action Items:**
- [ ] Create `Leak_detector/dashboard_app.py` based on latestMVP.py structure
- [ ] Add "Leak Detection Mode" UI components
- [ ] Integrate inference pipeline results display

---

### ⚠️ Category B: Needs Modification

#### 2.5 Anomaly Detection Logic
**File:** `latestMVP.py` (Isolation Forest implementation)
**Reusability:** 0% (different ML approach)

**Current Implementation:**
- Uses `sklearn.ensemble.IsolationForest` (unsupervised)
- Detects outliers/anomalies without labels
- Not suitable for binary classification (leak vs normal)

**What's Needed Instead:**
- **AI Pattern Recognition** based on actual waveform physics
- Detects HIGH sustained amplitude (smear pattern) = LEAK
- Outputs leak probability with confidence score

**Why Not Supervised ML:**
- Training data has only 8-10 unique valves (need 50+)
- Labels are inconsistent (same valve labeled differently)
- CSV statistics are misleading (don't match XML waveforms)

**Action Items:**
- [ ] Build `LeakDetector` class using AI pattern recognition in `Leak_detector/app/core/leak_detector.py`
- [ ] Analyze actual XML waveforms for smear vs spike patterns
- [ ] Defer supervised ML (XGBoost + RF) until proper training data available

---

### ❌ Category C: Build from Scratch

#### 2.6 AI Pattern Recognition Module
**Status:** Not found in existing codebase
**Required for:** Week 2 leak detection

**What's Needed:**
```python
class AILeakDetector:
    """AI-powered pattern recognition for valve leak detection"""

    def detect_leak(self, waveform: np.ndarray) -> LeakDetectionResult:
        """
        Analyzes waveform patterns:
        - Mean/median amplitude (HIGH sustained = leak)
        - Above threshold ratios (% above 1G, 2G, 5G)
        - Smear vs spike pattern recognition
        - Multi-feature weighted scoring

        Returns: probability, confidence, explanation
        """
```

**Why This Approach:**
- Training data has quality issues (inconsistent labels, only 8-10 unique valves)
- Pattern recognition based on actual physics is more reliable
- Explainable results (shows which patterns triggered detection)
- No training bias from mislabeled data

**Action Items:**
- [ ] Build `Leak_detector/app/core/leak_detector.py` from scratch (Week 2, Day 1-2)
- [ ] Implement waveform analysis based on actual XML patterns
- [ ] Validate against known leak files (C402 Cyl 3 CD, 578-B Cyl 3)

#### 2.7 Training Data Requirements Documentation
**Status:** Not found in existing codebase
**Required for:** Client communication, future ML enhancement

**What's Needed:**
```markdown
# Training Data Requirements for Supervised ML

## Minimum Requirements:
- 50+ unique leak valves (not repeat measurements)
- 50+ unique normal valves
- Consistent labeling (based on physical inspection)
- Metadata: timestamps, operating conditions, verification method
```

**Why Needed:**
- Current data insufficient for supervised ML (only 8-10 unique valves)
- Client needs clear guidance on what data to collect
- Foundation for future ML enhancement when proper data available

**Action Items:**
- [ ] Create `Leak_detector/docs/TRAINING_DATA_REQUIREMENTS.md` (Week 2, Day 3)
- [ ] Define data collection protocol for client
- [ ] Specify labeling criteria and metadata requirements

#### 2.8 Inference Pipeline
**Status:** Not found in existing codebase
**Required for:** Week 3 deliverable

**What's Needed:**
```python
class ValveLeakInference:
    """Production inference pipeline"""

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_extractor = AEFeatureExtractor()

    def predict_from_file(self, xml_file_path: str) -> Dict:
        """
        Returns:
        {
            'machine_id': str,
            'timestamp': str,
            'valves': [
                {'valve_id': str, 'leak_probability': float, 'status': str, 'confidence': str},
                ...
            ]
        }
        """
```

**Action Items:**
- [ ] Build `Leak_detector/app/inference/inference.py` (Week 3, Day 1-2)
- [ ] Integrate XML parser + feature extractor + trained model
- [ ] Add JSON/CSV output formatters

---

### 🔮 Category D: Future Phase (Phase 2)

#### 2.9 WRPM Parser
**File:** `app/core/wrpm_parser.py`
**Status:** Complete but not usable for pilot

**Why Deferred:**
- Test WRPM samples contain ONLY pressure data (no AE or vibration)
- Leak detection model requires AE data
- Cannot train on AE but run inference on files without AE (fundamental ML limitation)

**Recommendation:**
- Keep WRPM parser code available
- IF client can provide WRPM files with AE data: integrate in Phase 2 (2 weeks effort)
- IF not: WRPM support requires separate pressure-based model (5-6 weeks effort)

**Action Items:**
- [ ] Document WRPM limitation in client communication
- [ ] Copy `wrpm_parser.py` to `Leak_detector/app/core/` for future use
- [ ] Add placeholder in inference pipeline for WRPM format detection

---

## 3. Technology Stack Validation

### Python Packages (from `latestMVP.py`)

**Already in Use (Validated):**
- ✅ `streamlit` - Dashboard framework
- ✅ `pandas`, `numpy` - Data manipulation
- ✅ `plotly` - Visualizations
- ✅ `libsql_client` - Turso database
- ✅ `xml.etree.ElementTree` - XML parsing
- ✅ `sklearn` - Machine learning (has IsolationForest, also has XGBoost/RandomForest)

**Need to Add:**
- `xgboost` - Gradient boosting classifier
- `imbalanced-learn` - SMOTE for class balancing
- `tqdm` - Progress bars for training
- `joblib` - Model serialization
- `pytest` - Unit testing

**Already in `Leak_detector/requirements.txt`:** ✅ All packages listed above

---

## 4. Database Schema Assessment

### Current Schema (from `latestMVP.py`)

**Good Tables (Reusable):**
```sql
-- Session management
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    machine_id TEXT,
    rpm TEXT
);

-- Raw waveform storage
CREATE TABLE waveform_data (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    cylinder_name TEXT,
    curve_name TEXT,
    crank_angle REAL,
    data_value REAL,
    curve_type TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

-- Configuration storage
CREATE TABLE configs (
    machine_id TEXT PRIMARY KEY,
    contamination REAL DEFAULT 0.05,
    pressure_anom_limit INT DEFAULT 10,
    valve_anom_limit INT DEFAULT 5,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Alerts
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    machine_id TEXT,
    cylinder TEXT,
    severity TEXT,
    message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Tables Needing Modification:**
```sql
-- Extend analyses table
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    cylinder_name TEXT,
    curve_name TEXT,
    anomaly_count INTEGER,           -- Keep for backward compatibility
    threshold REAL,                  -- Keep
    leak_probability REAL,           -- NEW: Add for leak detection
    status TEXT,                     -- NEW: NORMAL / SUSPICIOUS / LEAK
    confidence TEXT,                 -- NEW: LOW / MEDIUM / HIGH
    model_version TEXT,              -- NEW: Track which model version
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
```

**New Tables Needed:**
```sql
-- Model metadata
CREATE TABLE model_metadata (
    id INTEGER PRIMARY KEY,
    version TEXT UNIQUE,
    model_type TEXT,                -- e.g., "XGBoost+RandomForest Ensemble"
    accuracy REAL,
    precision_leak REAL,
    recall_leak REAL,
    f1_score REAL,
    training_date DATETIME,
    feature_names TEXT,             -- JSON array of feature names
    training_samples_count INTEGER,
    notes TEXT
);

-- Predictions log
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    valve_id TEXT,
    cylinder_name TEXT,
    valve_type TEXT,                -- e.g., "HE Discharge 1 (US)"
    leak_probability REAL,
    status TEXT,                    -- NORMAL / SUSPICIOUS / LEAK
    confidence TEXT,                -- LOW / MEDIUM / HIGH
    model_version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

-- Training samples (for future model retraining)
CREATE TABLE training_samples (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    valve_id TEXT,
    label TEXT,                     -- LEAK / NORMAL
    features_json TEXT,             -- JSON-encoded feature vector
    waveform_ref TEXT,              -- Reference to waveform_data
    added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    validated BOOLEAN DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
```

**Action Items:**
- [ ] Create database migration script `Leak_detector/scripts/migrate_database.py`
- [ ] Test schema creation with Turso client
- [ ] Add indices for performance (session_id, valve_id, created_at)

---

## 5. Dashboard Components Analysis

### Reusable UI Components from `latestMVP.py`

**✅ Can Reuse Directly:**
1. **File Upload Section** (lines 432-550)
   - Multi-file uploader with validation
   - File size display
   - "Start New Analysis" button
   - Session state management

2. **Custom CSS Styling** (`style.css`)
   - Section headers
   - Color scheme
   - Responsive layouts

3. **Sidebar Configuration** (throughout)
   - Contamination factor sliders
   - Threshold controls
   - Machine selection dropdowns

4. **Progress Indicators**
   - Spinners during processing
   - Status messages (st.success, st.error, st.warning)

5. **Export Functionality** (report generation)
   - PDF generation (ReportLab)
   - CSV downloads
   - Database save

**⚠️ Needs Adaptation:**
1. **Analysis Results Display**
   - Current: Shows anomaly counts per valve
   - Needed: Show leak probability with color-coded status

2. **Health Score Calculation** (lines 100-186)
   - Current: Based on anomaly count thresholds
   - Needed: Based on leak probability classification

3. **Waveform Visualization**
   - Current: Highlights anomalies with red markers
   - Needed: Display leak probability overlay

**Action Items:**
- [ ] Create `Leak_detector/dashboard_app.py` based on latestMVP.py structure (Week 4, Day 1-2)
- [ ] Replace anomaly detection UI with leak detection UI
- [ ] Add color-coded valve status: 🟢 Normal | 🟡 Suspicious | 🔴 Leak
- [ ] Add leak probability gauges (Plotly indicator charts)

---

## 6. Gaps Analysis

### What's Missing from Existing Codebase

| Component | Status | Priority | Week | Effort |
|-----------|--------|----------|------|--------|
| AI Pattern Recognition System | ❌ Not found | High | Week 2 | 8 hours |
| Waveform Analysis Module | ❌ Not found | High | Week 2 | 5 hours |
| Training Data Requirements Doc | ❌ Not found | High | Week 2 | 3 hours |
| Inference Pipeline Script | ❌ Not found | High | Week 3 | 5 hours |
| JSON/CSV Output Formatter | ❌ Not found | Medium | Week 3 | 3 hours |
| Leak Detection Dashboard Mode | ⚠️ Partial | Medium | Week 4 | 4 hours |
| WRPM Integration (AE data) | 🔮 Deferred | Low | Phase 2 | N/A |

**Total Effort for Missing Components:** ~28 hours (within Week 2-4 budget of 37 hours)

### Critical Gap: Training Data Quality

**Not a missing component, but a fundamental data issue:**

| Issue | Impact | Resolution |
|-------|--------|------------|
| Only 8-10 unique valves | Model memorizes, doesn't generalize | AI pattern recognition (no training bias) |
| Inconsistent labels | Model learns contradictory patterns | Use physics-based detection |
| CSV/XML format mismatch | Misleading statistics | Analyze actual XML waveforms |

**Implication:** Supervised ML (XGBoost + RF) deferred until proper training data is available (50+ unique valves per class with consistent labeling).

---

## 7. Risk Assessment

### Low Risk ✅
- **XML Parsing:** Excellent module already exists, 100% compatible
- **Database Integration:** Turso client works, schema extensions straightforward
- **Dashboard Framework:** Streamlit UI patterns well-established
- **Visualization:** Plotly integration mature and tested

### Medium Risk ⚠️
- **Pattern Recognition Accuracy:** AI pattern recognition should achieve 90%+ confidence on known leaks
- **Data Quality Issues:** Training data has inconsistent labels and insufficient unique valves
- **Client Expectations:** Need to communicate why supervised ML is deferred

### High Risk 🔴
- **WRPM Format:** Cannot support WRPM for pilot if files lack AE data (confirmed limitation)
- **Training Data Quality:** Only 8-10 unique valves with inconsistent labels - cannot train reliable supervised ML
- **Client Understanding:** Client may have expected supervised ML; need to explain why AI pattern recognition is better given data limitations

### Mitigation Strategies
1. **Pattern Recognition:** Use AI pattern recognition based on actual waveform physics (validated on known leaks)
2. **Data Quality:** Don't train on inconsistent data; use physics-based approach instead
3. **Client Communication:** Document training data requirements clearly; explain path to supervised ML
4. **WRPM:** Document as Phase 2 enhancement; focus on XML for pilot
5. **Timeline:** Front-load AI pattern recognition development in Week 2

---

## 8. Recommendations

### Immediate Actions (Week 1)

1. ✅ **Copy XML Parser Module**
   ```bash
   cp ../phase1-modular/app/core/xml_parser.py Leak_detector/app/core/
   ```

2. ✅ **Copy Sample Data for Testing**
   ```bash
   cp ../assets/xml-samples/*.xml Leak_detector/assets/xml-samples/
   cp ../ml_training_dataset.csv Leak_detector/data/raw/
   ```

3. [ ] **Test Database Connectivity**
   ```python
   # Verify Turso credentials work
   python Leak_detector/scripts/test_database.py
   ```

4. [ ] **Validate XML Parsing on Sample Files**
   ```python
   # Ensure AE sensor data can be extracted
   from app.core.xml_parser import load_curves_data
   df, columns = load_curves_data(open('assets/xml-samples/578_A_Curves.xml').read())
   print(f"Found {len(columns)} sensors: {columns}")
   ```

### Week 2-4 Development Strategy

**Week 2: AI Pattern Recognition Development**
- Build waveform analysis based on actual XML patterns
- Reference database save/load patterns from `latestMVP.py`
- Create training data requirements document for client
- Validate pattern recognition on known leak files

**Week 3: Inference Pipeline**
- Reuse XML parser functions from Week 1
- Integrate AI pattern recognition from Week 2
- Follow dashboard structure from `latestMVP.py` for consistency

**Week 4: Dashboard Integration**
- Copy majority of dashboard UI from `latestMVP.py`
- Replace anomaly detection sections with AI leak detection
- Reuse file upload, export, and reporting infrastructure
- Display confidence scores and pattern explanations

---

## 9. Technology Stack Approval

### Validated Technologies

| Technology | Current Usage | Leak Detection Use | Status |
|------------|---------------|-------------------|---------|
| Python 3.11+ | ✅ In use | Core language | ✅ Approved |
| Streamlit | ✅ In use | Dashboard | ✅ Approved |
| Pandas/NumPy | ✅ In use | Data processing | ✅ Approved |
| Plotly | ✅ In use | Visualizations | ✅ Approved |
| Turso/libSQL | ✅ In use | Database | ✅ Approved |
| scikit-learn | ✅ In use | ML framework | ✅ Approved |
| XGBoost | ❌ Not in use | Binary classifier | ✅ Approved (add to requirements) |
| SMOTE | ❌ Not in use | Data augmentation | ✅ Approved (imbalanced-learn) |

**No Technology Risks Identified**

---

## 10. Next Steps

### Week 1 Completion
- [x] Infrastructure audit complete
- [ ] Data analysis (Day 3-4, next task)
- [ ] Week 2 strategy document (Day 5)

### Week 2 Kickoff Prerequisites
- [ ] Receive 80 additional leak samples from client
- [ ] Verify Turso database credentials
- [ ] Set up virtual environment with all dependencies
- [ ] Copy XML parser module to Leak_detector
- [ ] Create Week 2 git branch

---

## Conclusion

**Overall Assessment:** Infrastructure is 60-70% ready for leak detection pilot.

**Key Strengths:**
- Excellent XML parsing module (phase1-modular/app/core/xml_parser.py)
- Mature Streamlit dashboard framework
- Working Turso database integration
- Proven Plotly visualization patterns

**Key Gaps (Addressable in Weeks 2-4):**
- AI pattern recognition system (8 hours, Week 2)
- Training data requirements documentation (3 hours, Week 2)
- Inference pipeline (5 hours, Week 3)
- Dashboard adaptations (4 hours, Week 4)

**Critical Training Data Issues:**
- Only 8-10 unique valves in training data (need 50+ per class)
- Inconsistent labeling across measurement sessions
- CSV statistics don't match actual XML waveform behavior
- **Decision:** Use AI pattern recognition instead of supervised ML for pilot

**Critical Dependencies:**
- Training data requirements document provided to client
- Turso database credentials available for testing
- WRPM support deferred to Phase 2 (confirmed with client)

**Confidence Level:** High (80%) that Week 2-4 can be delivered on time using AI pattern recognition approach.

---

**Report Prepared By:** Andrea
**Date:** November 11, 2025 (Updated: November 17, 2025)
**Next Review:** Week 2 Start (November 18, 2025)
