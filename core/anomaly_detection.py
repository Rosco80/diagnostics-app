"""
Anomaly detection and rule-based diagnostics.

No Streamlit imports — pure data logic only.
"""

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def run_anomaly_detection(df, curve_names, contamination_level=0.05):
    """
    Applies Isolation Forest to detect anomalies, calculate severity scores,
    and normalize them into 0-1 confidence values.
    """
    for curve in curve_names:
        if curve in df.columns:
            data = df[[curve]].values
            model = IsolationForest(contamination=contamination_level, random_state=42)

            # Fit the model
            predictions = model.fit_predict(data)
            df[f'{curve}_anom'] = predictions == -1

            # Raw anomaly scores (lower = more anomalous)
            raw_scores = model.score_samples(data)

            # Flip so higher = more anomalous
            severity_scores = -1 * raw_scores
            df[f'{curve}_anom_score'] = severity_scores

            # Normalize scores into 0-1 confidence range
            scaler = MinMaxScaler(feature_range=(0, 1))
            confidences = scaler.fit_transform(severity_scores.reshape(-1, 1))
            df[f'{curve}_anom_confidence'] = confidences.flatten()

            # Optional: map into levels
            def classify_confidence(c):
                if c >= 0.8:
                    return "CRITICAL"
                elif c >= 0.6:
                    return "HIGH"
                elif c >= 0.4:
                    return "MEDIUM"
                else:
                    return "LOW"
            df[f'{curve}_anom_level'] = [classify_confidence(c) for c in confidences.flatten()]

    return df


def run_rule_based_diagnostics(report_data):
    """
    DEPRECATED: Fault type suggestions disabled - AI only detects anomalies.
    Users should manually classify faults using interactive tagging.
    Returns empty dict to maintain backward compatibility.
    """
    suggestions = {}  # Empty - no automatic fault classification
    return suggestions


def run_rule_based_diagnostics_enhanced(report_data, pressure_limit=10, valve_limit=5):
    """
    Enhanced diagnostics - Returns critical alerts based on anomaly counts.
    NOTE: Fault type suggestions disabled - AI only detects anomalies, users manually classify via tagging.
    """
    suggestions = {}  # Empty - no automatic fault classification
    critical_alerts = []

    for item in report_data:
        item_name = item['name']
        anomaly_count = item['count']

        # Pressure-specific alerts (counts only, no fault guessing)
        if item_name == 'Pressure':
            if anomaly_count > pressure_limit * 2:
                critical_alerts.append(f"CRITICAL: {item_name} has {anomaly_count} anomalies (limit: {pressure_limit})")
            elif anomaly_count > pressure_limit:
                critical_alerts.append(f"HIGH: {item_name} has {anomaly_count} anomalies (limit: {pressure_limit})")

        # Valve-specific alerts (counts only, no fault guessing)
        elif item_name != 'Pressure':
            if 'Suction' in item_name and anomaly_count > valve_limit * 2:
                critical_alerts.append(f"CRITICAL: {item_name} has {anomaly_count} anomalies (limit: {valve_limit})")
            elif 'Discharge' in item_name and anomaly_count > valve_limit * 1.5:
                critical_alerts.append(f"HIGH: {item_name} has {anomaly_count} anomalies (limit: {valve_limit})")
            elif anomaly_count > valve_limit:
                critical_alerts.append(f"WARNING: {item_name} has {anomaly_count} anomalies (limit: {valve_limit})")

    # Check for "all valves affected" condition (cylinder-level issue)
    he_valves = [item for item in report_data if 'HE' in item['name'] and item['name'] != 'Pressure']
    ce_valves = [item for item in report_data if 'CE' in item['name'] and item['name'] != 'Pressure']

    for valves, end_name in [(he_valves, 'Head End'), (ce_valves, 'Crank End')]:
        if len(valves) > 0:
            high_anomaly_count = sum(1 for v in valves if v['count'] > max(valve_limit * 2, 15))
            affected_percentage = (high_anomaly_count / len(valves)) * 100

            if high_anomaly_count / len(valves) >= 0.85:  # 85% or more valves severely affected
                critical_alerts.append(
                    f"CRITICAL: All valves on {end_name} affected ({affected_percentage:.0f}%) - cylinder-level issue suspected"
                )

    return suggestions, critical_alerts  # suggestions always empty now
