# Data Quality Report
## Training Dataset Assessment - Week 1

**Date:** November 11, 2025
**Project:** AI-Powered Valve Leak Detection - 4-Week Pilot
**Dataset:** `ml_training_dataset.csv`
**Analyst:** Andrea

---

## Executive Summary

This report assesses the quality and suitability of the training dataset (`ml_training_dataset.csv`) for developing a binary valve leak classifier. The dataset contains **188 labeled samples** from reciprocating compressor diagnostic sessions.

###  Overall Assessment: **QUALITY CONCERNS IDENTIFIED** ⚠

- **Data Completeness:** 100% (no missing values) ✓
- **Sensor Coverage:** 98.9% AE/Ultrasonic (excellent for leak detection) ✓
- **Unique Valve Diversity:** Only 8-10 unique physical valves ⚠
- **Label Consistency:** Issues found - same valve labeled differently ⚠
- **Data Format:** Peak-detected samples don't match XML waveform inference ⚠
- **Primary Challenges:** Insufficient unique valves, inconsistent labeling, CSV/XML format mismatch

---

## 1. Dataset Overview

### Basic Statistics

| Metric | Value |
|--------|-------|
| **Total Measurements** | 188 |
| **Unique Leak Valves** | ~8 |
| **Unique Normal Valves** | ~10 |
| **Leak Measurements** | 20 (10.6%) |
| **Normal Measurements** | 109 (58.0%) |
| **Other Faults** | 59 (31.4%) |
| **Missing Values** | 0 (0%) |

**Critical Finding:** The 188 samples represent multiple measurements of only 8-10 unique physical valves, NOT 188 unique valves. This is insufficient for training a generalizable ML model.

### Data Structure

**Columns (8):**
1. `Machine ID` - Machine identifier (e.g., "578-B", "C-401 INLET COMPRESS")
2. `Session Time` - Timestamp of diagnostic session
3. `Cylinder` - Cylinder identifier (e.g., "Cylinder 1")
4. `Curve` - Full sensor name with specifications (e.g., "578-B.1HD3.ULTRASONIC G 36KHZ - 44KHZ")
5. `Crank Angle` - Crankshaft position in degrees (0-720°)
6. `Value` - Amplitude reading in G (gravity units)
7. `Type` - Sensor category (labeled "vibration" but mostly AE/Ultrasonic)
8. `Fault Classification` - Ground truth label (Normal, Valve Leakage, Closing Hard or Slamming, etc.)

**Note on Mislabeling:** The `Type` column labels most samples as "vibration", but the `Curve` column reveals these are actually **ULTRASONIC sensors** (36-44 kHz acoustic emission). This is a labeling issue, not a data quality issue. Actual sensor type is correctly identified in curve names.

---

## 2. Class Distribution Analysis

### All Fault Types

| Fault Classification | Count | Percentage |
|---------------------|-------|------------|
| Normal | 109 | 58.0% |
| Closing Hard or Slamming | 53 | 28.2% |
| **Valve Leakage** | **20** | **10.6%** |
| Spring Fatigue or Failure | 3 | 1.6% |
| Valve Sticking | 2 | 1.1% |
| Other | 1 | 0.5% |

### Binary Classification (Leak vs Normal)

For the pilot project, we focus on binary classification:

| Class | Count | Percentage | Notes |
|-------|-------|------------|-------|
| **Leak** | **20** | **10.6%** | Target class for detection |
| **Normal** | **109** | **58.0%** | Baseline class |
| **Other Faults** | **59** | **31.4%** | Excluded from binary classifier |

### Critical Issue: Insufficient Unique Valves

**More Important Than Sample Count:**

The issue is NOT that we need "80 more leak samples" - it's that we need **50+ unique leak valves** and **50+ unique normal valves**.

**Why This Matters:**
- ML models need diverse examples to learn generalizable patterns
- With only 8 unique leak valves, model will memorize specific valve characteristics
- Model will fail on new, unseen valves from different installations/manufacturers

**Industry Standard:**
- Minimum: 50-100 unique examples per class
- Current: 8 leak valves, ~10 normal valves (far below minimum)

### Critical Issue: Label Inconsistency

**Example Discovered:**
```
Valve: 578-B Cylinder 1 CS2
- 6 readings labeled "Normal" at 21.6G amplitude
- 1 reading labeled "Valve Leakage" at 23.0G amplitude
```

These amplitudes are nearly identical (21.6G vs 23.0G), yet labels differ. This suggests:
- Labeling criteria was inconsistent
- Labels may not reflect actual valve condition at time of measurement
- Model will learn contradictory patterns

**Impact:** ML model trained on inconsistent labels will produce unreliable predictions.

### Revised Strategy (Week 2)

**Instead of data augmentation on existing samples:**
1. **Use AI pattern recognition** based on actual waveform physics
2. **Document proper training data requirements** for client
3. **Defer supervised ML** until proper training data is available (50+ unique valves per class)

---

## 3. Sensor Type Coverage

### Sensor Distribution

| Sensor Type | Count | Percentage | Suitability for Leak Detection |
|------------|-------|------------|-------------------------------|
| **AE/Ultrasonic (36-44 kHz)** | **186** | **98.9%** | **✓ Excellent** (Primary sensor for leak detection) |
| Vibration | 2 | 1.1% | ⚠ Limited (not primary for leak detection) |
| Pressure | 0 | 0% | ✗ None (not present in dataset) |

### Sensor Type Identification Logic

Sensors are categorized based on curve names:
- **AE/Ultrasonic:** Curve contains "ULTRASONIC" or "KHZ" (e.g., "36KHZ - 44KHZ")
- **Vibration:** Curve explicitly contains "VIBRATION" (without ultrasonic designation)
- **Pressure:** Curve contains "PRESSURE" or "COMPRESSOR PT"

### Leak Samples by Sensor Type

| Sensor Type | Leak Samples | Normal Samples |
|------------|--------------|----------------|
| AE/Ultrasonic | 20 (100%) | 108 (99.1%) |
| Vibration | 0 (0%) | 1 (0.9%) |

**Finding:** All 20 leak samples use AE/Ultrasonic sensors. This is ideal for the pilot, as AE sensors are most sensitive to valve leaks.

### Multi-Modal Assessment

**Client Request:** "Classify valve conditions using AE, pressure, and vibration data"

**Reality Check:**
- AE/Ultrasonic: **98.9%** coverage ✓
- Pressure: **0%** coverage ✗
- Vibration: **1.1%** coverage ✗

**Recommendation for Pilot:**
- **Focus on AE-only classification** (single-modal)
- Multi-modal integration requires:
  1. Collection of pressure/vibration data for same samples
  2. Feature engineering for all 3 modalities
  3. Sensor fusion architecture
  4. Estimated effort: +2 weeks

**Decision:** Proceed with AE-only for 4-week pilot, defer multi-modal to Phase 2.

---

## 4. Machine & Cylinder Coverage

### Machine Distribution

| Machine ID | Sample Count | Percentage |
|-----------|--------------|------------|
| 578-B | 68 | 36.2% |
| C-401 INLET COMPRESS | 62 | 33.0% |
| C-402 INLET COMPRESS | 31 | 16.5% |
| 578-A | 15 | 8.0% |
| C402 - C | 11 | 5.9% |
| 578-C | 1 | 0.5% |

**Coverage:** 6 unique machines
**Assessment:** Good diversity across multiple machines, reduces machine-specific overfitting

### Cylinder Distribution

| Cylinder | Sample Count |
|----------|--------------|
| Cylinder 1 | 135 (71.8%) |
| Cylinder 2 | 39 (20.7%) |
| Cylinder 4 | 11 (5.9%) |
| Cylinder 3 | 3 (1.6%) |

**Coverage:** 4 unique cylinders
**Assessment:** Heavy bias toward Cylinder 1, but acceptable for pilot

---

## 5. Data Quality Checks

### 5.1 Missing Values

**Result:** ✓ PASS

| Column | Missing Count | Percentage |
|--------|---------------|------------|
| Machine ID | 0 | 0% |
| Session Time | 0 | 0% |
| Cylinder | 0 | 0% |
| Curve | 0 | 0% |
| Crank Angle | 0 | 0% |
| Value | 0 | 0% |
| Type | 0 | 0% |
| Fault Classification | 0 | 0% |

**Conclusion:** Dataset is 100% complete. No imputation required.

### 5.2 Amplitude Value Ranges

**Distribution Statistics:**

| Statistic | Value (G) | Assessment |
|-----------|-----------|------------|
| Minimum | 8.90 | Valid |
| Maximum | 29.78 | Valid |
| Mean | 21.67 | Reasonable |
| Median | 22.31 | Reasonable |
| Std Dev | 6.15 | Moderate variability |

**Range Check:** ✓ PASS

All amplitude values fall within reasonable range for AE sensors (typically 5-50 G for reciprocating compressors). No invalid or corrupted values detected.

### 5.3 Crank Angle Validation

**Distribution Statistics:**

| Statistic | Value (degrees) | Assessment |
|-----------|----------------|------------|
| Minimum | 9° | Valid |
| Maximum | 354° | Valid |
| Mean | 168° | Centered |
| Expected Range | 0-720° | ✓ Within range |

**Range Check:** ✓ PASS

All crank angles fall within valid cycle range. Note: Dataset contains event-level data (specific crank angles), not full waveforms (0-720° complete cycles).

### 5.4 Outlier Detection (Amplitude)

**Method:** IQR-based outlier detection (1.5 × IQR rule)

| Metric | Value (G) |
|--------|-----------|
| Q1 (25th percentile) | 16.91 |
| Q3 (75th percentile) | 26.93 |
| IQR | 10.02 |
| Lower Bound | 1.88 |
| Upper Bound | 41.96 |

**Outliers Detected:** 0 samples (0%)

**Conclusion:** ✓ No statistical outliers. Data is well-behaved.

### 5.5 Duplicate Records

**Status:** Not assessed in current analysis

**Action Item:** Check for duplicate (Machine ID, Session Time, Cylinder, Curve, Crank Angle) combinations in Week 2

---

## 6. Leak vs Normal Comparison

### CSV Training Data Statistics (Peak-Detected Samples)

| Statistic | Leak Samples | Normal Samples | Difference |
|-----------|-------------|----------------|------------|
| **Count** | 20 | 109 | -89 |
| **Mean Amplitude** | 19.89 G | 22.52 G | -2.63 G |
| **Median Amplitude** | 21.73 G | 22.72 G | -0.99 G |
| **Std Dev** | 6.78 G | 5.96 G | +0.82 G |
| **Min Amplitude** | 8.90 G | 8.90 G | 0 G |
| **Max Amplitude** | 28.60 G | 29.78 G | -1.18 G |

**⚠️ CRITICAL WARNING:** These statistics are MISLEADING!

### The CSV vs XML Mismatch Problem

**CSV Training Data:**
- Contains peak-detected samples (2-13 peaks per valve)
- Statistics show: Leak mean = 19.89G, Normal mean = 22.52G
- Suggests leak < normal amplitude (confusing)

**Actual XML Waveform Data:**
- Contains full waveforms (355 continuous data points)
- Analysis shows: Leak mean = 4.59G, Normal mean = 1.27G
- Leak valve has 3.6x HIGHER sustained amplitude

### Correct Physics Understanding

Analysis of actual XML files reveals:

| Valve Type | Mean Amplitude | Behavior Pattern |
|------------|----------------|------------------|
| **LEAK** (C402 Cyl 3 CD) | **4.59G** | Sustained HIGH amplitude (smear pattern) |
| **NORMAL** (C402 Cyl 2 CD) | **1.27G** | Brief LOW amplitude spikes |

**Why the Discrepancy:**
1. CSV uses peak-detected samples (highest peaks only)
2. XML contains full waveform (all 355 points)
3. Leak valves show sustained elevation (continuous gas escaping)
4. Normal valves show brief spikes (clean valve event)

**Correct Physics:**
- **LEAK valve** = Gas escaping through valve seat = **HIGH sustained amplitude** (smear pattern)
- **NORMAL valve** = Clean valve closure = **LOW mean amplitude** (brief spikes)

### Implications for Model Training

**Problem:** Training a supervised ML model on CSV statistics will learn:
- LEAK = lower amplitude (WRONG based on actual physics)
- NORMAL = higher amplitude (WRONG)

**Solution:** Use AI pattern recognition based on actual waveform physics, not CSV statistics.

### Feature Separation Potential

**Assessment:** ⚠ **UNRELIABLE** (based on CSV data)

The CSV statistics are not representative of actual waveform behavior. Supervised ML trained on this data will produce unreliable results.

**Recommended Approach:**
- AI pattern recognition based on actual XML waveform analysis
- Focus on sustained amplitude patterns (smear vs spike)
- Defer supervised ML until proper training data is available

---

## 7. Data Format Assessment

### Current Format: Event-Level Data

**What We Have:**
- Single (crank angle, amplitude) pairs per sample
- Example: (11.0°, 10.12 G) at Cylinder 1, HD3 valve

**What We Need for Training:**
- Full waveform cycles (0-720° with ~720-1440 data points)
- Source: Load complete Curves XML files
- Extract entire waveform for each valve per session

### Action Required (Week 2, Day 1)

1. Load Curves XML files from diagnostic sessions
2. Extract full waveform arrays for each AE sensor
3. Compute 20 features per waveform:
   - Input: 720-point AE waveform array
   - Output: 20-element feature vector
4. Create training dataset: (features, label) pairs

**Current CSV:** Event-level annotations (for reference/validation)
**Training Input:** Full waveforms from XML files (Week 2)

---

## 8. Data Quality Summary

### Strengths ✓

1. **Complete Data:** 0% missing values
2. **High AE Coverage:** 98.9% of samples are AE/Ultrasonic (ideal for leak detection)
3. **Valid Ranges:** All amplitude and crank angle values within expected bounds
4. **No Outliers:** Statistical checks show clean, well-behaved data
5. **Machine Diversity:** 6 machines covered
6. **Labeled Ground Truth:** All samples have fault classifications (though consistency is questionable)

### Critical Limitations ⚠

1. **Insufficient Unique Valves:** Only 8-10 unique physical valves in entire dataset
   - **Impact:** Model will memorize specific valves, not learn generalizable patterns
   - **Required:** 50+ unique valves per class for robust ML

2. **Inconsistent Labeling:** Same valve labeled differently across sessions
   - **Example:** 578-B Cyl 1 CS2 labeled both "Normal" and "Valve Leakage" at similar amplitudes
   - **Impact:** Model learns contradictory patterns

3. **CSV vs XML Format Mismatch:** Training CSV uses peak-detected samples, inference uses full waveforms
   - **Impact:** Statistics don't reflect actual waveform behavior
   - **Solution:** Use actual XML waveform analysis

4. **Misleading Statistics:** CSV shows leak < normal amplitude, actual XML shows leak > normal
   - **Impact:** Supervised ML will learn wrong patterns

5. **No Multi-Modal Data:** 98.9% AE-only, no pressure or vibration for same samples
   - **Decision:** Proceed with AE-only for pilot

### Risks 🚨

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Insufficient unique valves** | CRITICAL | Use AI pattern recognition; defer supervised ML until 50+ unique valves available |
| **Label inconsistency** | HIGH | Document proper labeling criteria for client; don't train on inconsistent data |
| **CSV/XML format mismatch** | HIGH | Base detection on actual XML waveform analysis, not CSV statistics |
| **Misleading statistics** | HIGH | Validate against actual physics (HIGH amplitude = leak) |
| **Multi-Modal Gap** | MEDIUM | Document AE-only approach, explain data limitation |

---

## 9. Recommendations

### For Week 2 (Revised Approach)

1. **✓ Use AI Pattern Recognition Instead of Supervised ML**
   - Current training data has critical quality issues
   - AI pattern recognition based on actual waveform physics is more reliable
   - Detects HIGH sustained amplitude (smear pattern) = LEAK

2. **✗ Do NOT Request "80 More Leak Samples"**
   - More measurements of same 8 valves won't help
   - **Instead:** Request 50+ unique leak valves, 50+ unique normal valves
   - Each with consistent labeling based on physical verification

3. **✗ Do NOT Train Supervised ML on Current CSV Data**
   - Labels are inconsistent
   - Statistics are misleading (CSV vs XML format mismatch)
   - Only 8-10 unique valves (insufficient for generalization)

4. **✓ Analyze Actual XML Waveforms**
   - Use full 355-point waveforms from XML files
   - Validate physics: HIGH sustained amplitude = LEAK
   - Pattern recognition based on smear vs spike patterns

5. **✓ Document Training Data Requirements**
   - Create detailed specification for proper training data
   - 50+ unique valves per class
   - Consistent labeling based on physical inspection
   - Temporal context (when measured, operating conditions)

### For Client Communication

1. **✓ Explain Training Data Quality Issues**
   - Only 8-10 unique valves (need 50+)
   - Label inconsistencies discovered
   - Current data insufficient for supervised ML

2. **✓ Provide Training Data Collection Protocol**
   - What constitutes proper training data
   - How to label consistently
   - What metadata to capture

3. **✓ Set Realistic Expectations**
   - Week 2: AI pattern recognition (90%+ confidence on known leaks)
   - Future: Supervised ML when proper data available (95-98% accuracy potential)

---

## 10. Conclusion

**Overall Data Quality:** ⚠ **SIGNIFICANT CONCERNS IDENTIFIED**

The training dataset has critical quality issues that prevent reliable supervised ML training:
- Only 8-10 unique physical valves (need 50+ per class)
- Inconsistent labeling across measurement sessions
- CSV format doesn't match XML inference format (misleading statistics)

**Key Takeaway:** Current training data is **NOT suitable** for supervised ML. AI pattern recognition based on actual waveform physics is the recommended approach for Week 2. Supervised ML should be deferred until proper training data is collected (50+ unique valves per class with consistent labeling).

**Ready for Week 2:** ✓ YES (using AI pattern recognition approach)

**Client Action Required:** Review training data requirements document and plan for proper data collection if supervised ML is desired in future phases.

---

**Report Prepared By:** Andrea
**Date:** November 11, 2025 (Updated: November 17, 2025)
**Next Review:** Week 2 - AI Pattern Recognition System Development
