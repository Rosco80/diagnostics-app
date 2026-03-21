"""
PDF report generation using ReportLab.

Streamlit is used for st.warning / st.error feedback messages only.
"""

import io
import datetime
import streamlit as st

from core.health_scoring import generate_executive_summary

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None, suggestions=None, health_score=None, critical_alerts=None):
    """
    Enhanced PDF report generator with improved UI and executive summary.
    """
    if not REPORTLAB_AVAILABLE:
        st.warning("ReportLab not installed. PDF generation unavailable.")
        return None

    # Set default values
    if suggestions is None:
        suggestions = {}
    if health_score is None:
        health_score = 50.0
    if critical_alerts is None:
        critical_alerts = []

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    styles = getSampleStyleSheet()
    story = []

    # Enhanced custom styles for better formatting
    title_style = styles['Title']
    title_style.fontSize = 20
    title_style.spaceAfter = 20
    title_style.alignment = 1  # Center alignment
    title_style.textColor = colors.darkblue

    heading_style = styles['Heading2']
    heading_style.fontSize = 14
    heading_style.spaceAfter = 12
    heading_style.spaceBefore = 16
    heading_style.textColor = colors.darkblue
    heading_style.borderWidth = 1
    heading_style.borderColor = colors.lightgrey
    heading_style.borderPadding = 8
    heading_style.backColor = colors.lightgrey

    subheading_style = styles['Heading3']
    subheading_style.fontSize = 12
    subheading_style.spaceAfter = 8
    subheading_style.spaceBefore = 12
    subheading_style.textColor = colors.darkblue

    # Header with title and logo space
    story.append(Paragraph("MACHINE DIAGNOSTICS REPORT", title_style))
    story.append(Spacer(1, 20))

    # Basic info in a well-formatted table
    basic_info = [
        ['Machine ID:', machine_id, 'Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ['Cylinder:', cylinder_name, 'RPM:', str(rpm)],
    ]

    basic_table = Table(basic_info, colWidths=[80, 120, 80, 120])
    basic_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # First column bold
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),  # Third column bold
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.lightblue, colors.white])
    ]))
    story.append(basic_table)
    story.append(Spacer(1, 25))

    # EXECUTIVE SUMMARY SECTION with enhanced formatting
    executive_summary = generate_executive_summary(machine_id, cylinder_name, health_score, report_data, suggestions)

    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    story.append(Spacer(1, 10))

    # Status box with improved color coding and layout
    status_color = colors.red
    status_bg_color = colors.pink
    if executive_summary['overall_status'] in ['EXCELLENT', 'GOOD']:
        status_color = colors.green
        status_bg_color = colors.lightgreen
    elif executive_summary['overall_status'] == 'FAIR':
        status_color = colors.orange
        status_bg_color = colors.lightyellow
    elif executive_summary['overall_status'] == 'POOR':
        status_color = colors.orangered
        status_bg_color = colors.mistyrose

    # Executive summary table with better spacing
    exec_data = [
        ['Overall Status:', executive_summary['overall_status']],
        ['Health Score:', f"{executive_summary['health_score']:.1f}/100"],
        ['Total Anomalies:', str(executive_summary['total_anomalies'])],
        ['Critical Issues:', str(len(critical_alerts))]
    ]

    exec_table = Table(exec_data, colWidths=[150, 200])
    exec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),  # Status value bold
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (1, 0), (1, 0), status_color),  # Color code status
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, -1), status_bg_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 20))

    # Critical Issues section with better formatting
    if executive_summary['critical_issues']:
        story.append(Paragraph("Critical Issues Identified", subheading_style))
        for issue in executive_summary['critical_issues'][:5]:  # Limit to 5 issues
            story.append(Paragraph(f"• {issue}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Top Diagnostics in a structured format
    if executive_summary['top_diagnostics']:
        story.append(Paragraph("Key Diagnostic Findings", subheading_style))
        for finding in executive_summary['top_diagnostics']:
            story.append(Paragraph(f"• {finding}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Recommendations in a formatted box
    story.append(Paragraph("Recommendations", subheading_style))
    rec_data = [[f"• {rec}"] for rec in executive_summary['recommendations'][:4]]  # Limit to 4
    rec_table = Table(rec_data, colWidths=[450])
    rec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 15))

    # Next Actions in a formatted box
    story.append(Paragraph("Next Actions", subheading_style))
    action_data = [[f"• {action}"] for action in executive_summary['next_actions'][:4]]  # Limit to 4
    action_table = Table(action_data, colWidths=[450])
    action_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightcyan),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 25))

    # Chart section with better integration
    if chart_fig:
        story.append(Paragraph("Diagnostic Chart", heading_style))
        story.append(Spacer(1, 10))
        try:
            img_buffer = io.BytesIO()
            chart_fig.write_image(img_buffer, format='png', width=600, height=400, scale=2)
            img_buffer.seek(0)
            from reportlab.platypus import Image
            # Center the image
            img = Image(img_buffer, width=500, height=333)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 20))
        except Exception as e:
            story.append(Paragraph(f"Chart generation error: {str(e)}", styles['Normal']))
            story.append(Spacer(1, 15))

    # Detailed Health Report with improved table formatting
    if not health_report_df.empty:
        story.append(Paragraph("Detailed Health Report", heading_style))
        story.append(Spacer(1, 10))

        # Convert DataFrame to table data
        table_data = [health_report_df.columns.tolist()] + health_report_df.values.tolist()

        # Calculate column widths based on content
        num_cols = len(table_data[0])
        col_width = 450 / num_cols  # Distribute evenly

        # Create table with improved styling
        health_table = Table(table_data, colWidths=[col_width] * num_cols)
        health_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(health_table)
        story.append(Spacer(1, 20))

    # Anomaly Analysis Details with enhanced formatting
    if report_data:
        story.append(Paragraph("Anomaly Analysis Details", heading_style))
        story.append(Spacer(1, 10))

        anomaly_data = [['Component', 'Anomaly Count', 'Avg. Threshold', 'Unit', 'Status']]
        for item in report_data:
            status = "High" if item.get('count', 0) > 5 else "Normal"
            anomaly_data.append([
                item.get('name', 'Unknown'),
                str(item.get('count', 0)),
                f"{item.get('threshold', 0):.2f}",
                item.get('unit', ''),
                status
            ])

        anomaly_table = Table(anomaly_data, colWidths=[120, 80, 90, 60, 80])

        # Create style with conditional coloring for status column
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]

        # Add conditional coloring for high anomaly counts
        for i, item in enumerate(report_data, 1):
            if item.get('count', 0) > 5:
                table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.red))
                table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))
            else:
                table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.green))

        anomaly_table.setStyle(TableStyle(table_style))
        story.append(anomaly_table)
        story.append(Spacer(1, 25))

    # Professional footer with border
    footer_data = [[f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | AI-Powered Machine Diagnostics Analyzer v2.0"]]
    footer_table = Table(footer_data, colWidths=[500])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(footer_table)

    # Build PDF with improved error handling
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None


def generate_pdf_report_enhanced(machine_id, rpm, cylinder_name, report_data, health_report_df, chart_fig=None, suggestions=None, health_score=None, critical_alerts=None, tagged_events=None):
    """Enhanced PDF report generator with executive summary."""
    if not REPORTLAB_AVAILABLE:
        st.warning("ReportLab not installed. PDF generation unavailable.")
        return None

    if suggestions is None:
        suggestions = {}
    if health_score is None:
        health_score = 50.0
    if critical_alerts is None:
        critical_alerts = []

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = styles['Title']
    title_style.fontSize = 18
    title_style.spaceAfter = 30

    heading_style = styles['Heading2']
    heading_style.fontSize = 14
    heading_style.spaceAfter = 12
    heading_style.textColor = colors.darkblue

    # Title
    story.append(Paragraph("MACHINE DIAGNOSTICS REPORT", title_style))
    story.append(Spacer(1, 12))

    # Basic info table
    basic_info = [
        ['Machine ID:', machine_id],
        ['Cylinder:', cylinder_name],
        ['RPM:', rpm],
        ['Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]
    ]

    basic_table = Table(basic_info, colWidths=[100, 200])
    basic_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(basic_table)
    story.append(Spacer(1, 20))

    # EXECUTIVE SUMMARY
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))

    if health_score >= 85:
        overall_status = 'EXCELLENT'
        status_color = colors.green
    elif health_score >= 70:
        overall_status = 'GOOD'
        status_color = colors.green
    elif health_score >= 55:
        overall_status = 'FAIR'
        status_color = colors.orange
    elif health_score >= 40:
        overall_status = 'POOR'
        status_color = colors.orangered
    else:
        overall_status = 'CRITICAL'
        status_color = colors.red

    total_anomalies = sum(item.get('count', 0) for item in report_data)
    exec_data = [
        ['Overall Status:', overall_status],
        ['Health Score:', f"{health_score:.1f}/100"],
        ['Total Anomalies:', str(total_anomalies)],
        ['Critical Issues:', str(len(critical_alerts))]
    ]

    exec_table = Table(exec_data, colWidths=[120, 150])
    exec_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (1, 0), (1, 0), status_color),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 15))

    # Chart
    if chart_fig:
        try:
            img_buffer = io.BytesIO()
            chart_fig.write_image(img_buffer, format='png', width=800, height=500, scale=2)
            img_buffer.seek(0)
            from reportlab.platypus import Image
            story.append(Paragraph("Diagnostic Chart", heading_style))
            story.append(Image(img_buffer, width=500, height=312))
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"Chart could not be generated. Error: {str(e)}", styles['Normal']))

    # Tagged Events Section
    if tagged_events:
        story.append(Paragraph("Classified Fault Tags", heading_style))
        for tag in tagged_events:
            if isinstance(tag, dict):
                story.append(Paragraph(f"• **{tag['fault_classification']}** at {tag['angle']:.2f}°", styles['Normal']))
            else:
                # Handle legacy tags (just angles)
                story.append(Paragraph(f"• Tagged at: {tag:.2f}°", styles['Normal']))
        story.append(Spacer(1, 15))

    doc.build(story)
    buffer.seek(0)
    return buffer
