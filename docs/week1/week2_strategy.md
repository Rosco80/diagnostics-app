# Week 2 Strategy Document
## AI Pattern Recognition & Training Data Requirements

**Date:** November 11, 2025 (Updated: November 17, 2025)
**Project:** AI-Powered Valve Leak Detection - 4-Week Pilot
**Duration:** November 18-22, 2025 (15 hours)
**Payment:** $375 upon deliverable acceptance

---

## Objective

Develop **AI pattern recognition system** achieving **90%+ confidence** on known leak detection, based on actual waveform physics analysis. Document proper training data requirements for future supervised ML enhancement.

**Why This Approach (vs Supervised ML):**
- Training data has only 8-10 unique valves (need 50+)
- Labels are inconsistent (same valve labeled differently)
- CSV statistics are misleading (don't match actual XML waveforms)
- AI pattern recognition based on physics is more reliable

---

## Approach

### 1. AI Pattern Recognition System (Day 1-2, 8 hours)

**Input:** Full AE waveforms from XML files (355 data points per valve, 0-720° crank angle)
**Output:** Leak probability, confidence score, explainable pattern analysis

#### Core Physics Understanding

**Based on actual XML waveform analysis:**
- **LEAK valve** = Gas escaping through valve seat = **HIGH sustained amplitude** (smear pattern)
- **NORMAL valve** = Clean valve closure = **LOW mean amplitude** (brief spikes)

**Validated on Known Files:**
- C402 Cyl 3 CD (known leak): Mean = 4.59G, 92% above 2G = LEAK (smear pattern)
- C402 Cyl 2 CD (normal): Mean = 1.27G, 17% above 2G = NORMAL (spike pattern)

#### Pattern Recognition Features

**Amplitude Statistics:**
- Mean amplitude (HIGH sustained = leak)
- Median amplitude
- Max amplitude

**Smear Pattern Detection:**
- % of samples above 1G threshold
- % of samples above 2G threshold
- % of samples above 5G threshold

**Signal Continuity:**
- Sustained elevation ratio (leak = continuous, normal = discrete)
- Multi-feature weighted scoring

**Rationale:** Unlike supervised ML which relies on potentially mislabeled training data, pattern recognition uses actual waveform physics. Leak valves show characteristic "smear" patterns (continuous high amplitude from gas escaping), while normal valves show brief spikes (clean valve events).

**Implementation:** `Leak_detector/app/core/leak_detector.py`

---

### 2. XML Waveform Analysis Module (Day 2, 3 hours)

**Goal:** Parse and analyze actual XML waveform data

#### Capabilities

| Function | Description |
|----------|-------------|
| **parse_curves_xml()** | Extract 355-point waveforms from XML |
| **identify_ultrasonic_curves()** | Find 36-44 KHz AE sensor data |
| **compute_amplitude_statistics()** | Mean, median, max, std for waveform |
| **detect_smear_pattern()** | Calculate sustained elevation ratios |

**Key Insight:** CSV training data uses peak-detected samples (misleading statistics). XML waveforms show actual physics: HIGH sustained amplitude = LEAK.

**Implementation:** `Leak_detector/app/core/xml_parser.py`

---

### 3. Training Data Requirements Documentation (Day 3, 4 hours)

**Goal:** Document what proper training data looks like for future supervised ML

#### Document Contents

1. **Minimum Sample Requirements**
   - 50+ unique leak valves (not repeat measurements)
   - 50+ unique normal valves
   - Total 200+ measurements across diverse compressors

2. **Labeling Criteria**
   - Based on physical inspection/verification
   - Consistent criteria across all measurements
   - Document verification method (pressure test, visual inspection)

3. **Metadata Requirements**
   - Timestamp of measurement
   - Operating conditions (load, pressure, temperature)
   - Hours since last maintenance
   - Compressor model and configuration

4. **Data Collection Protocol**
   - Step-by-step guide for field engineers
   - Quality assurance checklist
   - How to handle edge cases

**Implementation:** `Leak_detector/docs/TRAINING_DATA_REQUIREMENTS.md`

---

### 4. Validation Testing (Day 4-5, 3 hours)

**Goal:** Validate AI pattern recognition on known leak examples

#### Test Cases

| File | Known Condition | Expected Result |
|------|----------------|-----------------|
| C402 Sep 9 1998 | Cyl 3 CD = LEAK | ≥90% leak probability |
| C402 Sep 9 1998 | Cyl 2 CD = NORMAL | ≤40% leak probability |
| 578-B Sep 25 2002 | Cyl 3 = LEAK | ≥50% leak probability |
| 578-A Sep 24 2002 | Normal operation | Low leak probabilities |

**Success Metric:** AI pattern recognition correctly identifies ALL known leaks with ≥90% confidence while maintaining <10% false positive rate on normal valves.

---

### 5. Evaluation Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| **Known Leak Detection** | ≥90% confidence | CRITICAL (safety) |
| **False Positive Rate** | <10% | HIGH |
| **Explainable Results** | Clear pattern explanation | HIGH |
| **Consistency** | Same input = same output | MEDIUM |

**Why Pattern Recognition is Better (for now):** Training on inconsistent labels will produce unreliable supervised ML. Physics-based pattern recognition using actual waveform analysis provides correct results immediately.

---

## Success Criteria

### Must-Have
- [ ] AI pattern recognition system complete and tested
- [ ] Known leak detection ≥90% confidence (C402 Cyl 3 CD)
- [ ] False positive rate <10% on normal valves
- [ ] Explainable results (pattern explanation for each detection)
- [ ] Training data requirements document complete
- [ ] Validation report showing correct classification of known leaks

### Nice-to-Have
- [ ] Detection confidence ≥95% on known leaks (stretch goal)
- [ ] Streamlit web interface for demo
- [ ] Performance comparison with supervised ML (showing why pattern recognition is better)

---

## Deliverables

1. **AI Pattern Recognition System** (`app/core/leak_detector.py`)
   - Waveform analysis based on actual physics
   - Detects HIGH sustained amplitude (smear) = LEAK
   - Provides confidence scores and explanations

2. **Training Data Requirements Document** (`docs/TRAINING_DATA_REQUIREMENTS.md`)
   - What data is needed for future supervised ML
   - 50+ unique valves per class
   - Labeling criteria and metadata requirements
   - Data collection protocol for client

3. **Source Code**
   - `app/core/leak_detector.py` - AI pattern recognition
   - `app/core/xml_parser.py` - Waveform extraction
   - `scripts/test_pattern_recognition.py` - Validation tests

4. **Validation Report** (2-3 pages)
   - Results on known leak files (C402, 578-B)
   - Confidence scores and pattern explanations
   - False positive analysis
   - Comparison with training data limitations

---

## Risk Mitigation

### Risk 1: Client Expects Supervised ML
**Mitigation:**
- Document why AI pattern recognition is better given data limitations
- Show training data quality issues (inconsistent labels, insufficient unique valves)
- Provide clear path to supervised ML when proper data is available
- Demonstrate 90%+ confidence on known leaks validates approach

### Risk 2: Pattern Recognition Doesn't Generalize
**Mitigation:**
- Base detection on actual physics (HIGH sustained amplitude = leak)
- Validate on multiple known leak files
- Adjust thresholds based on actual waveform analysis
- Document assumptions and edge cases

### Risk 3: Training Data Requirements Too Demanding
**Mitigation:**
- Provide clear justification (industry standards, ML fundamentals)
- Show current limitations (8 unique valves vs 50+ needed)
- Offer phased approach (AI pattern recognition now, supervised ML later)
- Include cost-benefit analysis of data collection effort

---

## Technical Stack

| Component | Tool/Library |
|-----------|-------------|
| Waveform Analysis | NumPy, Pandas, SciPy |
| XML Parsing | xml.etree.ElementTree |
| Pattern Recognition | Custom algorithms based on physics |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Documentation | Markdown |

---

## Timeline

| Day | Task | Hours |
|-----|------|-------|
| **Day 1** | AI pattern recognition system (core logic) | 4 |
| **Day 1** | XML waveform analysis module | 4 |
| **Day 2** | Validation testing on known leak files | 3 |
| **Day 3** | Training data requirements documentation | 4 |
| **Day 4** | Streamlit demo interface (optional) | 2 |
| **Day 5** | Performance report and final validation | 2 |
| **Total** | | **15-19 hours** |

---

## Dependencies

- **Week 1 Complete:** Infrastructure audit, data analysis ✓
- **Training Data Quality Issues:** Documented and understood ✓
- **Known Leak Files:** C402, 578-B XML samples available ✓
- **Environment:** Python 3.11, all dependencies installed
- **Data:** Access to Curves XML files for full waveforms

---

## Future Enhancement Path

**When Client Provides Proper Training Data:**

1. Collect 50+ unique leak valves with consistent labels
2. Collect 50+ unique normal valves with physical verification
3. Document metadata (timestamps, operating conditions)
4. Train supervised ML (XGBoost + RF ensemble)
5. Expected accuracy: 95-98% with proper data

**Current AI pattern recognition serves as production baseline until then.**

---

**Prepared By:** Andrea
**Date:** November 11, 2025 (Updated: November 17, 2025)
**Status:** Ready to Execute - Revised approach based on training data quality assessment
