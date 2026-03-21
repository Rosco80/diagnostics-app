"""
Health score computation and executive summary generation.

No Streamlit imports — pure data logic only.
"""


def compute_health_score(report_data, diagnostics):
    """
    Computes a simple health index from anomalies and rule-based diagnostics.
    Starts from 100 and subtracts a penalty for each anomaly and each diagnostic message.
    Returns a value between 0 and 100.
    """
    score = 100
    # Subtract one point per anomaly detected
    for item in report_data:
        score -= item['count'] * 0.2
    # Subtract an additional five points per diagnostic rule triggered
    score -= 5 * len(diagnostics)
    # Keep the score within 0-100
    return max(min(score, 100), 0)


def generate_executive_summary(machine_id, cylinder_name, health_score, report_data, suggestions):
    """
    Generates executive summary data for PDF reports.
    """
    summary = {
        'overall_status': 'UNKNOWN',
        'health_score': health_score,
        'total_anomalies': 0,
        'critical_issues': [],
        'top_diagnostics': [],
        'recommendations': [],
        'next_actions': []
    }

    # Calculate total anomalies
    total_anomalies = sum(item.get('count', 0) for item in report_data)
    summary['total_anomalies'] = total_anomalies

    # Determine overall status based on health score
    if health_score >= 85:
        summary['overall_status'] = 'EXCELLENT'
    elif health_score >= 70:
        summary['overall_status'] = 'GOOD'
    elif health_score >= 55:
        summary['overall_status'] = 'FAIR'
    elif health_score >= 40:
        summary['overall_status'] = 'POOR'
    else:
        summary['overall_status'] = 'CRITICAL'

    # Identify critical issues (high anomaly counts)
    for item in report_data:
        if item.get('count', 0) > 10:  # Threshold for critical
            summary['critical_issues'].append(f"{item['name']}: {item['count']} anomalies detected")

    # Get top 3 diagnostics from suggestions
    diagnostic_items = list(suggestions.items())[:3]
    for name, diagnosis in diagnostic_items:
        summary['top_diagnostics'].append(f"{name}: {diagnosis}")

    # Generate recommendations based on status
    if summary['overall_status'] == 'CRITICAL':
        summary['recommendations'] = [
            "Immediate maintenance attention required",
            "Consider equipment shutdown for inspection",
            "Schedule detailed valve inspection"
        ]
        summary['next_actions'] = [
            "Contact maintenance team within 24 hours",
            "Perform detailed diagnostic analysis",
            "Plan maintenance shutdown"
        ]
    elif summary['overall_status'] == 'POOR':
        summary['recommendations'] = [
            "Schedule maintenance within 1-2 weeks",
            "Monitor equipment closely",
            "Increase inspection frequency"
        ]
        summary['next_actions'] = [
            "Schedule maintenance inspection",
            "Continue monitoring",
            "Review operating parameters"
        ]
    elif summary['overall_status'] == 'FAIR':
        summary['recommendations'] = [
            "Schedule routine maintenance",
            "Monitor trending parameters",
            "Consider minor adjustments"
        ]
        summary['next_actions'] = [
            "Plan routine maintenance",
            "Track performance trends",
            "Optimize operating conditions"
        ]
    else:  # GOOD or EXCELLENT
        summary['recommendations'] = [
            "Continue current maintenance schedule",
            "Equipment operating within normal parameters",
            "Monitor for any trending changes"
        ]
        summary['next_actions'] = [
            "Maintain current schedule",
            "Continue routine monitoring",
            "No immediate action required"
        ]

    return summary
