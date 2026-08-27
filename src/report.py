from pathlib import Path
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# JOBGUARD PDF REPORT GENERATOR
# ============================================================


# ============================================================
# PAGE SETTINGS
# ============================================================

PAGE_WIDTH, PAGE_HEIGHT = A4

MARGIN_LEFT = 18 * mm
MARGIN_RIGHT = 18 * mm
MARGIN_TOP = 18 * mm
MARGIN_BOTTOM = 18 * mm


# ============================================================
# COLORS
# ============================================================

DARK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6B7280")
BORDER = colors.HexColor("#D1D5DB")
LIGHT_BG = colors.HexColor("#F9FAFB")

GREEN = colors.HexColor("#16A34A")
YELLOW = colors.HexColor("#CA8A04")
ORANGE = colors.HexColor("#EA580C")
RED = colors.HexColor("#DC2626")

WHITE = colors.white


# ============================================================
# RISK COLOR
# ============================================================

def get_risk_color(risk_level):

    if risk_level == "Low Risk":
        return GREEN

    if risk_level == "Medium Risk":
        return YELLOW

    if risk_level == "High Risk":
        return ORANGE

    return RED


# ============================================================
# RISK DESCRIPTION
# ============================================================

def get_risk_description(risk_score):

    if risk_score < 30:
        return (
            "The model estimates a relatively low fraud risk. "
            "Normal employer verification is still recommended."
        )

    if risk_score < 60:
        return (
            "The model identifies potentially suspicious "
            "characteristics. Review the employer and "
            "recruitment process carefully."
        )

    if risk_score < 80:
        return (
            "The model identifies a relatively high fraud risk. "
            "Independently verify the employer before proceeding."
        )

    return (
        "The model identifies a very high fraud risk. "
        "Exercise extreme caution before sharing personal "
        "or financial information."
    )


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(value):

    if value is None:
        return "Not provided"

    value = str(value).strip()

    if not value:
        return "Not provided"

    return value


# ============================================================
# ESCAPE HTML
# ============================================================

def escape_html(value):

    value = safe_value(value)

    return (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ============================================================
# COLOR HEX
# ============================================================

def color_hex(reportlab_color):

    return reportlab_color.hexval()


# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def draw_page(canvas, document):

    canvas.saveState()

    # --------------------------------------------------------
    # Header line
    # --------------------------------------------------------

    canvas.setStrokeColor(
        BORDER
    )

    canvas.line(
        MARGIN_LEFT,
        PAGE_HEIGHT - 12 * mm,
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - 12 * mm
    )


    # --------------------------------------------------------
    # Header left
    # --------------------------------------------------------

    canvas.setFont(
        "Helvetica-Bold",
        8
    )

    canvas.setFillColor(
        MUTED
    )

    canvas.drawString(
        MARGIN_LEFT,
        PAGE_HEIGHT - 9 * mm,
        "JOBGUARD"
    )


    # --------------------------------------------------------
    # Header right
    # --------------------------------------------------------

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - 9 * mm,
        "AI-Powered Job Scam Detection"
    )


    # --------------------------------------------------------
    # Footer line
    # --------------------------------------------------------

    canvas.line(
        MARGIN_LEFT,
        12 * mm,
        PAGE_WIDTH - MARGIN_RIGHT,
        12 * mm
    )


    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    canvas.setFont(
        "Helvetica",
        7
    )

    canvas.setFillColor(
        MUTED
    )

    canvas.drawString(
        MARGIN_LEFT,
        8 * mm,
        "JobGuard — Machine-learning risk assessment"
    )


    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        8 * mm,
        f"Page {document.page}"
    )


    canvas.restoreState()


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def generate_pdf_report(
    result,
    signals,
    job,
    analysis_time=None
):

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if analysis_time is None:

        analysis_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


    # --------------------------------------------------------
    # Result values
    # --------------------------------------------------------

    risk_score = float(
        result.get(
            "risk_score",
            0
        )
    )


    fraud_probability = float(
        result.get(
            "fraud_probability",
            0
        )
    )


    prediction = safe_value(
        result.get(
            "prediction_label"
        )
    )


    risk_level = safe_value(
        result.get(
            "risk_level"
        )
    )


    risk_color = get_risk_color(
        risk_level
    )


    risk_color_hex = color_hex(
        risk_color
    )


    # --------------------------------------------------------
    # PDF buffer
    # --------------------------------------------------------

    buffer = BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=MARGIN_RIGHT,

        leftMargin=MARGIN_LEFT,

        topMargin=25 * mm,

        bottomMargin=20 * mm,

        title="JobGuard Job Scam Risk Analysis",

        author="JobGuard"
    )


    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "JobGuardTitle",

        parent=styles["Title"],

        fontName="Helvetica-Bold",

        fontSize=24,

        leading=29,

        textColor=DARK,

        alignment=TA_CENTER,

        spaceAfter=5
    )


    subtitle_style = ParagraphStyle(

        "JobGuardSubtitle",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=10,

        leading=14,

        textColor=MUTED,

        alignment=TA_CENTER,

        spaceAfter=18
    )


    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontName="Helvetica-Bold",

        fontSize=14,

        leading=18,

        textColor=DARK,

        spaceBefore=10,

        spaceAfter=8
    )


    normal_style = ParagraphStyle(

        "NormalJobGuard",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=9.5,

        leading=14,

        textColor=DARK
    )


    small_style = ParagraphStyle(

        "Small",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=8,

        leading=11,

        textColor=MUTED
    )


    label_style = ParagraphStyle(

        "Label",

        parent=styles["Normal"],

        fontName="Helvetica-Bold",

        fontSize=8.5,

        leading=11,

        textColor=MUTED
    )


    value_style = ParagraphStyle(

        "Value",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=9,

        leading=12,

        textColor=DARK
    )


    risk_heading_style = ParagraphStyle(

        "RiskHeading",

        parent=normal_style,

        fontName="Helvetica-Bold",

        fontSize=11,

        leading=14,

        textColor=risk_color
    )


    # ========================================================
    # STORY
    # ========================================================

    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )


    story.append(
        Paragraph(
            "JOBGUARD",
            title_style
        )
    )


    story.append(
        Paragraph(
            "JOB SCAM RISK ANALYSIS REPORT",
            subtitle_style
        )
    )


    story.append(
        Paragraph(
            f"<b>Analysis Date:</b> "
            f"{escape_html(analysis_time)}",
            small_style
        )
    )


    story.append(
        Spacer(
            1,
            6 * mm
        )
    )


    # ========================================================
    # RISK SUMMARY
    # ========================================================

    risk_table_data = [

        [
            Paragraph(
                "<b>RISK SCORE</b>",
                label_style
            ),

            Paragraph(
                "<b>PREDICTION</b>",
                label_style
            ),

            Paragraph(
                "<b>FRAUD PROBABILITY</b>",
                label_style
            ),

            Paragraph(
                "<b>RISK LEVEL</b>",
                label_style
            )
        ],

        [
            Paragraph(
                f"<font size='22'><b>"
                f"{risk_score:.1f}/100"
                f"</b></font>",
                normal_style
            ),

            Paragraph(
                escape_html(
                    prediction
                ),
                normal_style
            ),

            Paragraph(
                f"{fraud_probability:.2%}",
                normal_style
            ),

            Paragraph(
                f"<font color='{risk_color_hex}'>"
                f"<b>{escape_html(risk_level)}</b>"
                f"</font>",
                normal_style
            )
        ]
    ]


    risk_table = Table(

        risk_table_data,

        colWidths=[
            40 * mm,
            40 * mm,
            48 * mm,
            42 * mm
        ]
    )


    risk_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BG
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    BORDER
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )


    story.append(
        risk_table
    )


    story.append(
        Spacer(
            1,
            5 * mm
        )
    )


    # ========================================================
    # RISK INTERPRETATION
    # ========================================================

    interpretation_data = [

        [
            Paragraph(
                escape_html(
                    risk_level
                ),
                risk_heading_style
            )
        ],

        [
            Paragraph(
                escape_html(
                    get_risk_description(
                        risk_score
                    )
                ),
                normal_style
            )
        ]
    ]


    interpretation_table = Table(

        interpretation_data,

        colWidths=[
            170 * mm
        ]
    )


    interpretation_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8FAFC")
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    BORDER
                ),

                (
                    "LINEBEFORE",
                    (0, 0),
                    (0, -1),
                    5,
                    risk_color
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    12
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    12
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                )
            ]
        )
    )


    story.append(
        interpretation_table
    )


    # ========================================================
    # JOB INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Job Information",
            section_style
        )
    )


    job_rows = [

        [
            Paragraph(
                "Job Title",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("title")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Location",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("location")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Department",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("department")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Employment Type",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("employment_type")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Experience",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("required_experience")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Education",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("required_education")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Industry",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("industry")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Job Function",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("function")
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Salary Range",
                label_style
            ),

            Paragraph(
                escape_html(
                    job.get("salary_range")
                ),
                value_style
            )
        ]
    ]


    job_table = Table(

        job_rows,

        colWidths=[
            45 * mm,
            125 * mm
        ]
    )


    job_table.setStyle(
        TableStyle(
            [

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    BORDER
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    LIGHT_BG
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )


    story.append(
        job_table
    )


    # ========================================================
    # RISK SIGNALS
    # ========================================================

    story.append(
        Paragraph(
            "2. Detected Risk Signals",
            section_style
        )
    )


    if signals:

        signal_rows = [

            [
                Paragraph(
                    "Severity",
                    label_style
                ),

                Paragraph(
                    "Risk Signal",
                    label_style
                ),

                Paragraph(
                    "Explanation",
                    label_style
                )
            ]
        ]


        for signal in signals:

            severity = safe_value(
                signal.get(
                    "severity"
                )
            )


            signal_name = safe_value(
                signal.get(
                    "signal"
                )
            )


            reason = safe_value(
                signal.get(
                    "reason"
                )
            )


            if severity in [
                "Very High",
                "High"
            ]:

                severity_color = RED

            elif severity == "Medium":

                severity_color = ORANGE

            elif severity == "Low":

                severity_color = YELLOW

            else:

                severity_color = GREEN


            severity_hex = color_hex(
                severity_color
            )


            severity_text = (
                f"<font color='{severity_hex}'>"
                f"<b>{escape_html(severity)}</b>"
                f"</font>"
            )


            signal_rows.append(

                [

                    Paragraph(
                        severity_text,
                        value_style
                    ),

                    Paragraph(
                        escape_html(
                            signal_name
                        ),
                        value_style
                    ),

                    Paragraph(
                        escape_html(
                            reason
                        ),
                        value_style
                    )
                ]
            )


        signal_table = Table(

            signal_rows,

            colWidths=[
                30 * mm,
                52 * mm,
                88 * mm
            ],

            repeatRows=1
        )


        signal_table.setStyle(
            TableStyle(
                [

                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.6,
                        BORDER
                    ),

                    (
                        "INNERGRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        BORDER
                    ),

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        LIGHT_BG
                    ),

                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),

                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),

                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )


        story.append(
            signal_table
        )

    else:

        story.append(
            Paragraph(
                "No specific risk signals were detected.",
                normal_style
            )
        )


    # ========================================================
    # SAFETY RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "3. Safety Recommendations",
            section_style
        )
    )


    recommendations = [

        "Verify the company's official website.",

        "Check that the recruiter's email uses a legitimate company domain.",

        "Never pay money to obtain a job.",

        "Never share passwords, OTPs or authentication codes.",

        "Never provide banking credentials to a recruiter.",

        "Verify unusually high salary claims.",

        "Research the company and recruiter independently.",

        "Be cautious of urgent hiring pressure."
    ]


    recommendation_rows = []


    for recommendation in recommendations:

        recommendation_rows.append(

            [

                Paragraph(
                    "✓",
                    ParagraphStyle(
                        "Check",
                        parent=normal_style,
                        textColor=GREEN,
                        fontSize=11
                    )
                ),

                Paragraph(
                    escape_html(
                        recommendation
                    ),
                    normal_style
                )
            ]
        )


    recommendation_table = Table(

        recommendation_rows,

        colWidths=[
            10 * mm,
            160 * mm
        ]
    )


    recommendation_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BG
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ]
        )
    )


    story.append(
        recommendation_table
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "4. Model Information",
            section_style
        )
    )


    model_name = safe_value(
        result.get(
            "model_name",
            "Logistic Regression"
        )
    )


    threshold = result.get(
        "threshold",
        None
    )


    model_rows = [

        [
            Paragraph(
                "Model",
                label_style
            ),

            Paragraph(
                escape_html(
                    model_name
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Decision Threshold",
                label_style
            ),

            Paragraph(
                (
                    f"{float(threshold):.2f}"
                    if threshold is not None
                    else "Configured in model"
                ),
                value_style
            )
        ],

        [
            Paragraph(
                "Risk Score",
                label_style
            ),

            Paragraph(
                f"{risk_score:.1f}/100",
                value_style
            )
        ],

        [
            Paragraph(
                "Fraud Probability",
                label_style
            ),

            Paragraph(
                f"{fraud_probability:.2%}",
                value_style
            )
        ]
    ]


    model_table = Table(

        model_rows,

        colWidths=[
            55 * mm,
            115 * mm
        ]
    )


    model_table.setStyle(
        TableStyle(
            [

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    BORDER
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    LIGHT_BG
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )


    story.append(
        model_table
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "5. Disclaimer",
            section_style
        )
    )


    disclaimer = (
        "JobGuard provides a machine-learning-based risk "
        "assessment. A prediction is not definitive proof "
        "that a job posting is fraudulent or legitimate. "
        "Users should independently verify the employer, "
        "recruiter, employment offer and communication "
        "channels before sharing sensitive information "
        "or accepting employment."
    )


    disclaimer_table = Table(

        [
            [
                Paragraph(
                    escape_html(
                        disclaimer
                    ),
                    small_style
                )
            ]
        ],

        colWidths=[
            170 * mm
        ]
    )


    disclaimer_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#FFF7ED")
                ),

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#FDBA74")
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                )
            ]
        )
    )


    story.append(
        disclaimer_table
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(

        story,

        onFirstPage=draw_page,

        onLaterPages=draw_page
    )


    # ========================================================
    # RETURN PDF DATA
    # ========================================================

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# TEST PDF GENERATION
# ============================================================

if __name__ == "__main__":

    print(
        "Testing JobGuard PDF report generator..."
    )


    test_result = {

        "risk_score": 25,

        "fraud_probability": 0.18,

        "prediction_label": "Legitimate",

        "risk_level": "Low Risk",

        "model_name": "Logistic Regression",

        "threshold": 0.70
    }


    test_signals = [

        {
            "severity": "Low",

            "signal": "Detailed job description",

            "reason": (
                "The posting contains substantial "
                "job-related information."
            )
        },

        {
            "severity": "Low",

            "signal": "Company information available",

            "reason": (
                "Company profile information is present."
            )
        }
    ]


    test_job = {

        "title": "Software Engineer",

        "location": "Hyderabad, Telangana, India",

        "department": "Engineering",

        "employment_type": "Full-time",

        "required_experience": "2-3 years",

        "required_education": "Bachelor's Degree",

        "industry": "Information Technology",

        "function": "Software Engineering",

        "salary_range": "Not specified"
    }


    try:

        pdf_data = generate_pdf_report(

            result=test_result,

            signals=test_signals,

            job=test_job
        )


        output_path = (
            Path(__file__).resolve().parent.parent
            / "JobGuard_Test_Report.pdf"
        )


        with open(
            output_path,
            "wb"
        ) as file:

            file.write(
                pdf_data
            )


        print()
        print(
            "PDF report generated successfully!"
        )

        print(
            f"File: {output_path}"
        )

        print(
            f"Size: {len(pdf_data):,} bytes"
        )

        print()


    except Exception as error:

        print()
        print(
            "PDF generation failed:"
        )

        print(
            error
        )

        print()

        raise