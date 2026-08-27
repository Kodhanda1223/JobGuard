import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
MODEL_DIR = SRC_DIR / "models"

MODEL_PATH = MODEL_DIR / "jobguard_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "jobguard_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "jobguard_metadata.joblib"


# ============================================================
# LOAD TRAINED JOBGUARD COMPONENTS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"JobGuard model not found: {MODEL_PATH}"
    )

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"JobGuard preprocessor not found: "
        f"{PREPROCESSOR_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"JobGuard metadata not found: "
        f"{METADATA_PATH}"
    )


model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

metadata = joblib.load(
    METADATA_PATH
)


# ============================================================
# MODEL THRESHOLD
# ============================================================

BEST_THRESHOLD = float(
    metadata["threshold"]
)


# ============================================================
# SALARY PARSER
# ============================================================

def parse_salary_range(salary_range):
    """
    Convert salary text into:
        salary_min
        salary_max
        salary_midpoint

    Examples:
        50000-70000
        50,000-70,000
        60000
    """

    if salary_range is None:
        return np.nan, np.nan, np.nan

    salary_text = str(
        salary_range
    ).strip()

    if salary_text == "":
        return np.nan, np.nan, np.nan

    salary_text = salary_text.replace(
        ",",
        ""
    )

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        salary_text
    )

    numbers = [
        float(number)
        for number in numbers
    ]

    if len(numbers) >= 2:

        salary_min = numbers[0]
        salary_max = numbers[1]

        salary_midpoint = (
            salary_min + salary_max
        ) / 2

        return (
            salary_min,
            salary_max,
            salary_midpoint
        )

    if len(numbers) == 1:

        salary_min = numbers[0]
        salary_max = numbers[0]

        return (
            salary_min,
            salary_max,
            salary_min
        )

    return np.nan, np.nan, np.nan


# ============================================================
# PREPARE JOB INPUT
# ============================================================

def prepare_job_input(
    title="",
    location="",
    department="",
    salary_range="",
    company_profile="",
    description="",
    requirements="",
    benefits="",
    employment_type="",
    required_experience="",
    required_education="",
    industry="",
    function="",
    telecommuting=0,
    has_company_logo=0,
    has_questions=0
):
    """
    Convert raw job-posting information into
    the feature structure expected by JobGuard.
    """

    # --------------------------------------------------------
    # SAFE TEXT CONVERSION
    # --------------------------------------------------------

    title = (
        "" if title is None
        else str(title)
    )

    location = (
        "" if location is None
        else str(location)
    )

    department = (
        "" if department is None
        else str(department)
    )

    company_profile = (
        "" if company_profile is None
        else str(company_profile)
    )

    description = (
        "" if description is None
        else str(description)
    )

    requirements = (
        "" if requirements is None
        else str(requirements)
    )

    benefits = (
        "" if benefits is None
        else str(benefits)
    )

    employment_type = (
        "" if employment_type is None
        else str(employment_type)
    )

    required_experience = (
        "" if required_experience is None
        else str(required_experience)
    )

    required_education = (
        "" if required_education is None
        else str(required_education)
    )

    industry = (
        "" if industry is None
        else str(industry)
    )

    function = (
        "" if function is None
        else str(function)
    )


    # --------------------------------------------------------
    # SALARY FEATURES
    # --------------------------------------------------------

    (
        salary_min,
        salary_max,
        salary_midpoint
    ) = parse_salary_range(
        salary_range
    )


    # --------------------------------------------------------
    # MISSINGNESS FEATURES
    # --------------------------------------------------------

    location_was_missing = int(
        location.strip() == ""
    )

    salary_was_missing = int(
        salary_range is None
        or str(salary_range).strip() == ""
        or np.isnan(salary_midpoint)
    )

    company_profile_was_missing = int(
        company_profile.strip() == ""
    )

    description_was_missing = int(
        description.strip() == ""
    )

    requirements_was_missing = int(
        requirements.strip() == ""
    )

    benefits_was_missing = int(
        benefits.strip() == ""
    )


    # --------------------------------------------------------
    # TEXT STATISTICS
    # --------------------------------------------------------

    description_length = len(
        description
    )

    description_word_count = len(
        description.split()
    )

    requirements_length = len(
        requirements
    )

    requirements_word_count = len(
        requirements.split()
    )

    benefits_length = len(
        benefits
    )

    benefits_word_count = len(
        benefits.split()
    )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    job_input = {

        # Location is retained for application use.
        # If the trained preprocessor doesn't use it,
        # ColumnTransformer will drop it.
        "location": location,

        # Salary
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_midpoint": salary_midpoint,

        # Description statistics
        "description_length": description_length,
        "description_word_count": (
            description_word_count
        ),

        # Requirements statistics
        "requirements_length": requirements_length,
        "requirements_word_count": (
            requirements_word_count
        ),

        # Benefits statistics
        "benefits_length": benefits_length,
        "benefits_word_count": (
            benefits_word_count
        ),

        # Binary features
        "telecommuting": int(
            telecommuting
        ),

        "has_company_logo": int(
            has_company_logo
        ),

        "has_questions": int(
            has_questions
        ),

        # Missingness indicators
        "description_was_missing": (
            description_was_missing
        ),

        "company_profile_was_missing": (
            company_profile_was_missing
        ),

        "requirements_was_missing": (
            requirements_was_missing
        ),

        "benefits_was_missing": (
            benefits_was_missing
        ),

        "location_was_missing": (
            location_was_missing
        ),

        "salary_was_missing": (
            salary_was_missing
        ),

        # Categorical features
        "employment_type": employment_type,

        "required_experience": (
            required_experience
        ),

        "required_education": (
            required_education
        ),

        "industry": industry,

        "function": function,

        "department": department,

        # Text features
        "title": title,

        "description": description,

        "requirements": requirements,

        "benefits": benefits
    }

    return pd.DataFrame(
        [job_input]
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(risk_score):
    """
    Convert a 0-100 risk score into a
    human-readable risk level.
    """

    if risk_score < 30:
        return "Low Risk"

    if risk_score < 60:
        return "Medium Risk"

    if risk_score < 80:
        return "High Risk"

    return "Very High Risk"


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_job(
    title="",
    location="",
    department="",
    salary_range="",
    company_profile="",
    description="",
    requirements="",
    benefits="",
    employment_type="",
    required_experience="",
    required_education="",
    industry="",
    function="",
    telecommuting=0,
    has_company_logo=0,
    has_questions=0
):
    """
    Run a complete JobGuard fraud-risk prediction.

    Returns:
        fraud_probability
        risk_score
        risk_level
        prediction
        prediction_label
    """

    # --------------------------------------------------------
    # 1. Prepare input
    # --------------------------------------------------------

    job_df = prepare_job_input(
        title=title,
        location=location,
        department=department,
        salary_range=salary_range,
        company_profile=company_profile,
        description=description,
        requirements=requirements,
        benefits=benefits,
        employment_type=employment_type,
        required_experience=required_experience,
        required_education=required_education,
        industry=industry,
        function=function,
        telecommuting=telecommuting,
        has_company_logo=has_company_logo,
        has_questions=has_questions
    )


    # --------------------------------------------------------
    # 2. Apply preprocessing
    # --------------------------------------------------------

    processed_job = (
        preprocessor.transform(
            job_df
        )
    )


    # --------------------------------------------------------
    # 3. Get fraud probability
    # --------------------------------------------------------

    fraud_probability = float(
        model.predict_proba(
            processed_job
        )[0, 1]
    )


    # --------------------------------------------------------
    # 4. Convert to 0-100 score
    # --------------------------------------------------------

    risk_score = round(
        fraud_probability * 100,
        1
    )


    # --------------------------------------------------------
    # 5. Determine risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        risk_score
    )


    # --------------------------------------------------------
    # 6. Classification using trained threshold
    # --------------------------------------------------------

    prediction = int(
        fraud_probability
        >= BEST_THRESHOLD
    )


    if prediction == 1:
        prediction_label = "Fraudulent"
    else:
        prediction_label = "Legitimate"


    # --------------------------------------------------------
    # 7. Return result
    # --------------------------------------------------------

    return {
        "fraud_probability": round(
            fraud_probability,
            4
        ),

        "risk_score": risk_score,

        "risk_level": risk_level,

        "prediction": prediction,

        "prediction_label": prediction_label
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_result = predict_job(
        title="Software Engineer",
        location="Hyderabad, India",
        department="Engineering",
        salary_range="600000-900000",
        company_profile=(
            "Established technology company "
            "building enterprise software."
        ),
        description=(
            "We are looking for a software "
            "engineer to join our development team."
        ),
        requirements=(
            "Experience with Python, SQL and "
            "software development."
        ),
        benefits=(
            "Health insurance, paid leave "
            "and professional development."
        ),
        employment_type="Full-time",
        required_experience="2-3 years",
        required_education="Bachelor's Degree",
        industry="Technology",
        function="Engineering",
        telecommuting=0,
        has_company_logo=1,
        has_questions=1
    )

    print()
    print("=" * 55)
    print("             JOBGUARD TEST")
    print("=" * 55)

    print(
        "Prediction        :",
        test_result["prediction_label"]
    )

    print(
        "Fraud Probability :",
        f"{test_result['fraud_probability']:.2%}"
    )

    print(
        "Risk Score        :",
        f"{test_result['risk_score']}/100"
    )

    print(
        "Risk Level        :",
        test_result["risk_level"]
    )

    print("=" * 55)

# ============================================================
# RISK SIGNAL EXPLANATION
# ============================================================

def get_risk_signals(
    title="",
    location="",
    salary_range="",
    company_profile="",
    description="",
    requirements="",
    benefits="",
    employment_type="",
    required_experience="",
    required_education="",
    industry="",
    function=""
):
    """
    Identify observable risk signals in a job posting.

    These signals are explanatory indicators and are
    NOT direct proof of fraud.
    """

    signals = []

    title_text = str(title).strip()
    location_text = str(location).strip()
    salary_text = str(salary_range).strip()
    company_text = str(company_profile).strip()
    description_text = str(description).strip()
    requirements_text = str(requirements).strip()
    benefits_text = str(benefits).strip()
    experience_text = str(required_experience).strip()
    education_text = str(required_education).strip()


    # --------------------------------------------------------
    # Missing information
    # --------------------------------------------------------

    if not company_text:
        signals.append({
            "signal": "Missing company profile",
            "severity": "Medium",
            "reason": (
                "The posting does not provide "
                "company information."
            )
        })


    if not location_text:
        signals.append({
            "signal": "Missing job location",
            "severity": "Low",
            "reason": (
                "The posting does not clearly "
                "specify where the job is located."
            )
        })


    if not salary_text:
        signals.append({
            "signal": "Missing salary information",
            "severity": "Low",
            "reason": (
                "No salary range was provided."
            )
        })


    if not requirements_text:
        signals.append({
            "signal": "Missing requirements",
            "severity": "Medium",
            "reason": (
                "The posting provides limited "
                "information about candidate requirements."
            )
        })


    if not benefits_text:
        signals.append({
            "signal": "Missing benefits information",
            "severity": "Low",
            "reason": (
                "Benefits or compensation details "
                "are not clearly described."
            )
        })


    # --------------------------------------------------------
    # Description quality
    # --------------------------------------------------------

    if len(description_text.split()) < 30:
        signals.append({
            "signal": "Very short job description",
            "severity": "Medium",
            "reason": (
                "The job description contains "
                "relatively little information."
            )
        })


    if len(requirements_text.split()) < 10:
        signals.append({
            "signal": "Limited candidate requirements",
            "severity": "Low",
            "reason": (
                "The requirements section contains "
                "limited detail."
            )
        })


    # --------------------------------------------------------
    # Suspicious language
    # --------------------------------------------------------

    suspicious_phrases = [
        "no experience required",
        "work from home",
        "make money fast",
        "easy money",
        "guaranteed income",
        "guaranteed salary",
        "earn money",
        "quick cash",
        "urgent hiring",
        "immediate joining",
        "no interview",
        "registration fee",
        "application fee",
        "pay to apply",
        "investment required",
        "send money",
        "whatsapp",
        "telegram"
    ]


    combined_text = (
        title_text + " "
        + company_text + " "
        + description_text + " "
        + requirements_text + " "
        + benefits_text
    ).lower()


    found_phrases = []

    for phrase in suspicious_phrases:

        if phrase in combined_text:

            found_phrases.append(
                phrase
            )


    if found_phrases:

        signals.append({
            "signal": "Potentially suspicious language",
            "severity": "High",
            "reason": (
                "The posting contains potentially "
                "concerning phrases: "
                + ", ".join(found_phrases[:5])
            )
        })


    # --------------------------------------------------------
    # Salary-related signals
    # --------------------------------------------------------

    salary_numbers = re.findall(
        r"\d+(?:,\d+)*(?:\.\d+)?",
        salary_text
    )


    if salary_numbers:

        try:

            salary_values = [
                float(
                    value.replace(",", "")
                )
                for value in salary_numbers
            ]

            if max(salary_values) > 10000000:

                signals.append({
                    "signal": "Unusually high salary figure",
                    "severity": "Medium",
                    "reason": (
                        "The supplied salary figure "
                        "appears unusually high and "
                        "should be independently verified."
                    )
                })

        except ValueError:
            pass


    # --------------------------------------------------------
    # Contact / payment signals
    # --------------------------------------------------------

    payment_phrases = [
        "pay a fee",
        "pay fee",
        "processing fee",
        "registration fee",
        "security deposit",
        "deposit money",
        "pay money"
    ]


    found_payment_phrases = [
        phrase
        for phrase in payment_phrases
        if phrase in combined_text
    ]


    if found_payment_phrases:

        signals.append({
            "signal": "Possible payment request",
            "severity": "Very High",
            "reason": (
                "The posting appears to reference "
                "payments or fees. Legitimate employers "
                "normally should not require candidates "
                "to pay money to obtain employment."
            )
        })


    # --------------------------------------------------------
    # Contact platform signals
    # --------------------------------------------------------

    if "telegram" in combined_text:

        signals.append({
            "signal": "Telegram mentioned",
            "severity": "Medium",
            "reason": (
                "The posting references Telegram. "
                "Independently verify the employer "
                "before communicating or sharing information."
            )
        })


    if "whatsapp" in combined_text:

        signals.append({
            "signal": "WhatsApp mentioned",
            "severity": "Low",
            "reason": (
                "The posting references WhatsApp. "
                "Verify the recruiter and company independently."
            )
        })


    # --------------------------------------------------------
    # No signals
    # --------------------------------------------------------

    if not signals:

        signals.append({
            "signal": "No obvious input-level risk signals",
            "severity": "Informational",
            "reason": (
                "No major risk indicators were detected "
                "from the supplied information. "
                "Independent verification is still recommended."
            )
        })


    return signals