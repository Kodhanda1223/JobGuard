from __future__ import annotations

import io
import html
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================================================
# APP / PATH CONFIG
# ============================================================

st.set_page_config(
    page_title="JobGuard | AI Job Scam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = SRC_DIR / "models"
MODEL_PATH = MODEL_DIR / "jobguard_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "jobguard_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "jobguard_metadata.joblib"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
<style>
:root { color-scheme: dark; }
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 90% 5%, rgba(79,70,229,.16), transparent 25%),
        radial-gradient(circle at 5% 85%, rgba(14,165,233,.08), transparent 28%),
        #070b14;
    color: #f8fafc;
}
.block-container { max-width: 1480px; padding-top: 1.5rem; padding-bottom: 4rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a1120 0%,#070c16 100%);
    border-right: 1px solid #1d2a3e;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.stButton > button, .stDownloadButton > button {
    min-height: 42px; border-radius: 11px; border: 1px solid #334155;
    background: #111827; color: #f8fafc; font-weight: 700;
    transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #6366f1; background: #182236; transform: translateY(-1px);
}
button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#2563eb) !important;
    border: 0 !important; box-shadow: 0 8px 24px rgba(79,70,229,.25);
}
.stTextInput input, .stTextArea textarea,
.stNumberInput input, div[data-baseweb="select"] > div {
    background: #0e1626 !important; color: #f8fafc !important;
    border-color: #2d3b52 !important; border-radius: 10px !important;
}
textarea { line-height: 1.55 !important; }
[data-testid="stDataFrame"] { border: 1px solid #26344b; border-radius: 12px; overflow: hidden; }
.jg-brand { display:flex; align-items:center; gap:12px; margin-bottom:26px; }
.jg-logo {
    width:48px;height:48px;border-radius:15px;display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#6366f1,#2563eb);font-size:25px;
    box-shadow:0 10px 30px rgba(79,70,229,.3);
}
.jg-brand-name { font-size:1.45rem;font-weight:850;color:#fff;line-height:1.1; }
.jg-brand-sub { font-size:.72rem;color:#64748b;margin-top:3px; }
.jg-eyebrow { color:#60a5fa;font-size:.72rem;font-weight:850;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px; }
.jg-title { font-size:2.85rem;font-weight:900;letter-spacing:-1.8px;color:#fff;line-height:1.05; }
.jg-subtitle { color:#94a3b8;font-size:1rem;margin-top:8px; }
.hero {
    background: radial-gradient(circle at 90% 15%,rgba(99,102,241,.22),transparent 32%),
                linear-gradient(135deg,#101c35,#0b1324);
    border:1px solid #2a426a;border-radius:22px;padding:32px;margin:22px 0 26px;
    box-shadow:0 18px 55px rgba(0,0,0,.18);
}
.hero h1 { margin:0;font-size:2.15rem;font-weight:850;letter-spacing:-1px; }
.hero p { max-width:900px;color:#a7b5ca;line-height:1.75;margin:10px 0 0; }
.section-title { font-size:1.45rem;font-weight:850;margin:30px 0 8px;color:#fff; }
.section-sub { color:#71809a;margin-bottom:14px; }
.card {
    background:linear-gradient(145deg,rgba(15,23,42,.98),rgba(9,14,26,.98));
    border:1px solid #23334d;border-radius:17px;padding:20px;margin-bottom:16px;
    box-shadow:0 12px 35px rgba(0,0,0,.14);
}
.metric-card { background:linear-gradient(145deg,#101a2c,#0c1321);border:1px solid #263650;border-radius:15px;padding:18px;min-height:125px; }
.metric-label { color:#8ea0b8;font-size:.72rem;text-transform:uppercase;font-weight:800;letter-spacing:1px; }
.metric-value { color:#f8fafc;font-size:1.65rem;font-weight:900;margin-top:9px; }
.metric-note { color:#64748b;font-size:.76rem;margin-top:5px; }
.pill { display:inline-block;padding:6px 10px;border-radius:999px;background:#172033;border:1px solid #2b3a54;color:#93c5fd;font-size:.72rem;font-weight:800;margin-right:5px; }
.notice { border-radius:13px;padding:14px 16px;border:1px solid #29415e;background:rgba(59,130,246,.07);color:#b8d7ff; }
.warning-box { border-radius:14px;padding:16px;border:1px solid rgba(245,158,11,.3);background:rgba(245,158,11,.07);color:#fbbf24; }
.success-box { border-radius:14px;padding:16px;border:1px solid rgba(16,185,129,.3);background:rgba(16,185,129,.07);color:#6ee7b7; }
.risk-banner { border-radius:17px;padding:21px;border:1px solid #263650;background:#0d1525; }
.small-muted { color:#64748b;font-size:.78rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA / DEFAULTS
# ============================================================

DEFAULT_JOB = {
    "title": "", "location": "", "department": "", "salary_range": "",
    "company_profile": "", "description": "", "requirements": "", "benefits": "",
    "employment_type": "", "required_experience": "", "required_education": "",
    "industry": "", "function": "", "telecommuting": 0,
    "has_company_logo": 1, "has_questions": 0,
}

DEMO_JOBS = {
    "Low-Risk Example": {
        **DEFAULT_JOB,
        "title":"Software Engineer", "location":"Hyderabad, Telangana, India",
        "department":"Engineering", "salary_range":"600000-900000",
        "company_profile":"Established technology company building enterprise software solutions for customers across India and international markets.",
        "description":"We are looking for a Software Engineer to join our engineering team. You will design, develop, test and maintain scalable software applications. You will work with product managers, developers and QA engineers using modern software development practices.",
        "requirements":"Bachelor's degree in Computer Science or related field. Knowledge of Python, Java or similar programming languages. Understanding of databases, REST APIs, Git and software development lifecycle. Strong problem-solving and communication skills.",
        "benefits":"Competitive salary, health insurance, paid leave, learning budget and professional development opportunities.",
        "employment_type":"Full-time", "required_experience":"Mid-level", "required_education":"Bachelor's Degree",
        "industry":"Information Technology", "function":"Engineering", "telecommuting":0, "has_company_logo":1, "has_questions":1,
    },
    "Medium-Risk Example": {
        **DEFAULT_JOB,
        "title":"Remote Data Entry Associate", "location":"Remote", "department":"Operations", "salary_range":"300000-450000",
        "company_profile":"Growing online services organization seeking detail-oriented professionals for operational support.",
        "description":"Work remotely entering and validating business information. Successful candidates should be comfortable working independently and communicating through online tools.",
        "requirements":"Basic computer skills, internet access, attention to detail and willingness to work flexible hours.",
        "benefits":"Flexible working schedule and performance-based incentives.", "employment_type":"Full-time",
        "required_experience":"Entry level", "required_education":"High School", "industry":"Administrative", "function":"Operations",
        "telecommuting":1, "has_company_logo":1, "has_questions":0,
    },
    "High-Risk Example": {
        **DEFAULT_JOB,
        "title":"Work From Home Data Entry - Immediate Hiring", "location":"Remote", "department":"", "salary_range":"80000-120000",
        "company_profile":"",
        "description":"URGENT!!! Work from home and earn guaranteed income of 80000 per month. No experience required. Immediate joining. Selected candidates can start today. Limited vacancies. Send your personal details immediately to secure your position.",
        "requirements":"No experience required. Anyone can apply. Act immediately. Candidates must be ready to complete verification.",
        "benefits":"Guaranteed income. Fast promotion. Work from anywhere. Immediate payment.", "employment_type":"Full-time",
        "required_experience":"Not Applicable", "required_education":"Not Applicable", "industry":"Other", "function":"Administrative",
        "telecommuting":1, "has_company_logo":0, "has_questions":0,
    },
}

PROCESSED_COLUMNS = [
    "job_id","title","location","department","salary_range","company_profile","description","requirements","benefits",
    "telecommuting","has_company_logo","has_questions","employment_type","required_experience","required_education","industry","function",
    "fraudulent","description_length","description_word_count","money_payment","bank_financial","urgent_action","guaranteed_income",
    "description_was_missing","company_profile_was_missing","location_was_missing","salary_was_missing","salary_min","salary_max","salary_midpoint",
    "requirements_was_missing","requirements_length","requirements_word_count","benefits_was_missing","benefits_length","benefits_word_count",
    "department_was_missing","employment_type_was_missing","required_experience_was_missing","required_education_was_missing","industry_was_missing","function_was_missing",
]

# ============================================================
# SESSION STATE — ALL WIDGET KEYS INITIALIZED BEFORE WIDGETS
# ============================================================

if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "imported_data" not in st.session_state: st.session_state.imported_data = None
if "demo_loaded" not in st.session_state: st.session_state.demo_loaded = False
if "page" not in st.session_state: st.session_state.page = "Analyze Job"

for _name, _value in DEFAULT_JOB.items():
    _key = f"field_{_name}"
    if _key not in st.session_state:
        st.session_state[_key] = _value

# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model_assets():
    errors, mdl, prep, meta = [], None, None, {}
    try:
        if MODEL_PATH.exists(): mdl = joblib.load(MODEL_PATH)
        else: errors.append(f"Model not found: {MODEL_PATH}")
    except Exception as exc: errors.append(f"Model loading error: {exc}")
    try:
        if PREPROCESSOR_PATH.exists(): prep = joblib.load(PREPROCESSOR_PATH)
        else: errors.append(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    except Exception as exc: errors.append(f"Preprocessor loading error: {exc}")
    try:
        if METADATA_PATH.exists(): meta = joblib.load(METADATA_PATH)
    except Exception as exc: errors.append(f"Metadata loading error: {exc}")
    return mdl, prep, meta, errors

model, preprocessor, metadata, model_errors = load_model_assets()

# ============================================================
# HELPERS / FEATURE ENGINEERING
# ============================================================

def clean_text(value):
    if value is None: return ""
    if isinstance(value, float) and np.isnan(value): return ""
    return str(value).strip()

def safe_int(value, default=0):
    try: return int(value)
    except Exception: return default

def text_stats(text):
    text = clean_text(text)
    return len(text), len(re.findall(r"\b\w+\b", text))

def parse_salary(value):
    value = clean_text(value)
    if not value: return np.nan, np.nan, np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
    if not nums: return np.nan, np.nan, np.nan
    nums = [float(x) for x in nums]
    if len(nums) == 1: return nums[0], nums[0], nums[0]
    low, high = min(nums), max(nums)
    return low, high, (low + high) / 2

def keyword_flag(text, keywords):
    text = clean_text(text).lower()
    return int(any(k.lower() in text for k in keywords))

def build_processed_row(job):
    job = {**DEFAULT_JOB, **job}
    title, location, department = clean_text(job["title"]), clean_text(job["location"]), clean_text(job["department"])
    salary = clean_text(job["salary_range"]); company = clean_text(job["company_profile"])
    description, requirements, benefits = clean_text(job["description"]), clean_text(job["requirements"]), clean_text(job["benefits"])
    employment, experience = clean_text(job["employment_type"]), clean_text(job["required_experience"])
    education, industry, function = clean_text(job["required_education"]), clean_text(job["industry"]), clean_text(job["function"])
    desc_len, desc_words = text_stats(description); req_len, req_words = text_stats(requirements); ben_len, ben_words = text_stats(benefits)
    salary_min, salary_max, salary_mid = parse_salary(salary)
    all_text = " ".join([title, description, requirements, benefits, company])
    row = {
        "job_id":0,"title":title,"location":location,"department":department,"salary_range":salary,"company_profile":company,
        "description":description,"requirements":requirements,"benefits":benefits,
        "telecommuting":safe_int(job["telecommuting"]),"has_company_logo":safe_int(job["has_company_logo"]),"has_questions":safe_int(job["has_questions"]),
        "employment_type":employment,"required_experience":experience,"required_education":education,"industry":industry,"function":function,"fraudulent":0,
        "description_length":desc_len,"description_word_count":desc_words,
        "money_payment":keyword_flag(all_text,["pay money","payment required","registration fee","processing fee","deposit","pay upfront","fee required","money transfer"]),
        "bank_financial":keyword_flag(all_text,["bank account","bank details","credit card","debit card","financial information","routing number"]),
        "urgent_action":keyword_flag(all_text,["urgent","immediately","act now","limited vacancies","today","asap","hurry"]),
        "guaranteed_income":keyword_flag(all_text,["guaranteed income","guaranteed salary","guaranteed earnings","earn guaranteed","fixed income"]),
        "description_was_missing":int(not description),"company_profile_was_missing":int(not company),"location_was_missing":int(not location),"salary_was_missing":int(not salary),
        "salary_min":salary_min,"salary_max":salary_max,"salary_midpoint":salary_mid,
        "requirements_was_missing":int(not requirements),"requirements_length":req_len,"requirements_word_count":req_words,
        "benefits_was_missing":int(not benefits),"benefits_length":ben_len,"benefits_word_count":ben_words,
        "department_was_missing":int(not department),"employment_type_was_missing":int(not employment),"required_experience_was_missing":int(not experience),
        "required_education_was_missing":int(not education),"industry_was_missing":int(not industry),"function_was_missing":int(not function),
    }
    return pd.DataFrame([row], columns=PROCESSED_COLUMNS)

SCAM_KEYWORDS = {
    "money_payment":["registration fee","processing fee","pay upfront","deposit","payment required","pay money","fee required"],
    "bank_financial":["bank account","bank details","credit card","debit card","financial information"],
    "urgent_action":["urgent","immediately","act now","limited vacancies","hurry","today","asap"],
    "guaranteed_income":["guaranteed income","guaranteed salary","guaranteed earnings","earn guaranteed"],
}

def heuristic_probability(job):
    text = " ".join(clean_text(job.get(k)) for k in ["title","description","requirements","benefits","company_profile"]).lower()
    score = sum(18 for keywords in SCAM_KEYWORDS.values() if any(k in text for k in keywords))
    if not clean_text(job.get("company_profile")): score += 10
    if not clean_text(job.get("salary_range")): score += 5
    if safe_int(job.get("has_company_logo")) == 0: score += 8
    if safe_int(job.get("has_questions")) == 0: score += 2
    if safe_int(job.get("telecommuting")) == 1: score += 2
    if len(clean_text(job.get("description"))) < 120: score += 5
    if "no experience" in text: score += 7
    if "work from home" in text and ("guaranteed" in text or "immediate" in text): score += 8
    return min(score / 100, .995)

def get_risk_level(score):
    if score < 30: return "Low Risk", "low", "🟢"
    if score < 60: return "Medium Risk", "medium", "🟡"
    if score < 80: return "High Risk", "high", "🟠"
    return "Very High Risk", "very-high", "🔴"

def generate_signals(job):
    combined = " ".join(clean_text(job.get(k)) for k in ["title","description","requirements","benefits","company_profile"]).lower()
    signals = []
    def add(severity, signal, reason): signals.append({"severity":severity,"signal":signal,"reason":reason})
    if any(x in combined for x in ["registration fee","processing fee","pay upfront","deposit","payment required","pay money"]):
        add("Critical","Payment or fee language","The posting contains language suggesting the candidate may be asked to pay money.")
    if any(x in combined for x in ["bank account","bank details","credit card","debit card","financial information"]):
        add("High","Financial information request","The posting appears to request sensitive financial information.")
    if any(x in combined for x in ["urgent","immediately","act now","limited vacancies","hurry","asap"]):
        add("Medium","Urgency language","Strong urgency can pressure applicants into acting before verifying the employer.")
    if any(x in combined for x in ["guaranteed income","guaranteed salary","guaranteed earnings","earn guaranteed"]):
        add("High","Guaranteed income claim","Unusually strong guaranteed-income claims can be a warning sign.")
    if not clean_text(job.get("company_profile")): add("Medium","Missing company profile","The employer provides little or no company background information.")
    if not clean_text(job.get("salary_range")): add("Low","Salary information missing","Compensation details are not clearly provided.")
    if safe_int(job.get("has_company_logo")) == 0: add("Low","No company logo","The posting does not contain a company logo.")
    if len(clean_text(job.get("description"))) < 120: add("Medium","Very short job description","The posting provides relatively little information about the role.")
    if not signals: add("Low","No major observable warning signals","No major rule-based warning indicators were detected in the supplied fields.")
    return signals

def predict_job(job):
    probability, source, model_error = None, "Trained ML model", None
    processed = build_processed_row(job)
    if model is not None and preprocessor is not None:
        try:
            X = preprocessor.transform(processed)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                probability = float(p[0,1] if p.shape[1] >= 2 else p[0,0])
            else:
                probability = float(model.predict(X)[0])
            probability = max(0.0, min(1.0, probability))
        except Exception as exc:
            model_error = str(exc)
    if probability is None:
        probability = heuristic_probability(job)
        source = "Fallback risk rules"
    score = probability * 100
    level, _, icon = get_risk_level(score)
    threshold = float(metadata.get("threshold", .7)) if isinstance(metadata, dict) else .7
    return {
        "probability":probability,"risk_score":score,"risk_level":level,"risk_icon":icon,
        "prediction":"Fraudulent" if probability >= threshold else "Legitimate",
        "signals":generate_signals(job),"source":source,"model_error":model_error,"threshold":threshold,
    }

# ============================================================
# IMPORT PARSERS
# ============================================================

def parse_txt_job(text):
    text = clean_text(text)
    if not text: return {}
    result = {}
    patterns = {
        "title":r"(?:job title|title)\s*[:\-]\s*(.+)","location":r"(?:location|job location)\s*[:\-]\s*(.+)",
        "salary_range":r"(?:salary|salary range|compensation)\s*[:\-]\s*(.+)","employment_type":r"(?:employment type|type)\s*[:\-]\s*(.+)",
        "industry":r"industry\s*[:\-]\s*(.+)","department":r"department\s*[:\-]\s*(.+)",
        "required_experience":r"(?:required experience|experience)\s*[:\-]\s*(.+)","required_education":r"(?:required education|education)\s*[:\-]\s*(.+)",
    }
    for line in text.splitlines():
        for field, pattern in patterns.items():
            match = re.search(pattern, line.strip(), flags=re.I)
            if match: result[field] = match.group(1).strip(); break
    result["description"] = text
    result["requirements"] = result.get("requirements") or text
    result["benefits"] = result.get("benefits", "")
    result["company_profile"] = result.get("company_profile", "")
    low = text.lower()
    result["telecommuting"] = int("remote" in low or "work from home" in low)
    result["has_company_logo"] = 1; result["has_questions"] = 0
    return {**DEFAULT_JOB, **result}

def normalize_column(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def parse_csv_row(row):
    normalized = {normalize_column(k):v for k,v in row.items()}
    aliases = {
        "title":["title","job_title","position"],"location":["location","job_location"],"salary_range":["salary_range","salary","compensation"],
        "company_profile":["company_profile","company_description","company_info"],"description":["description","job_description"],
        "requirements":["requirements","required_skills","skills"],"benefits":["benefits"],"employment_type":["employment_type","employment"],
        "required_experience":["required_experience","experience"],"required_education":["required_education","education"],
        "industry":["industry"],"department":["department"],"function":["function","job_function"],
        "telecommuting":["telecommuting","remote"],"has_company_logo":["has_company_logo","company_logo"],"has_questions":["has_questions","screening_questions"],
    }
    job = DEFAULT_JOB.copy()
    for target, keys in aliases.items():
        for key in keys:
            if key in normalized:
                value = normalized[key]
                if isinstance(value, float) and np.isnan(value): value = ""
                job[target] = value; break
    for key in ["telecommuting","has_company_logo","has_questions"]:
        v = job[key]
        if isinstance(v,str): job[key] = int(v.strip().lower() in {"1","true","yes","y","remote","on"})
        else: job[key] = int(bool(v))
    return job

def parse_csv_file(uploaded):
    df = pd.read_csv(uploaded)
    if df.empty: return df, None
    return df, parse_csv_row(df.iloc[0].to_dict())

# ============================================================
# SAFE FORM CALLBACKS
# ============================================================

def load_demo_callback():
    selected = st.session_state.get("demo_selector", "Low-Risk Example")
    job = DEMO_JOBS.get(selected, DEMO_JOBS["Low-Risk Example"])
    for key, value in job.items(): st.session_state[f"field_{key}"] = value
    st.session_state.demo_loaded = True
    st.session_state.last_result = None
    st.session_state.imported_data = None

def clear_form_callback():
    for key, value in DEFAULT_JOB.items(): st.session_state[f"field_{key}"] = value
    st.session_state.demo_loaded = False
    st.session_state.imported_data = None
    st.session_state.last_result = None

def load_imported_callback():
    job = st.session_state.get("imported_data") or {}
    for key, value in job.items():
        if key not in DEFAULT_JOB: continue
        if isinstance(value,float) and np.isnan(value): value = ""
        if key in {"telecommuting","has_company_logo","has_questions"}:
            value = int(value) if isinstance(value,(int,np.integer)) else int(str(value).strip().lower() in {"1","true","yes","y","remote","on"})
        st.session_state[f"field_{key}"] = value
    st.session_state.demo_loaded = True
    st.session_state.last_result = None

def save_history(job, result):
    st.session_state.history.insert(0, {
        "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title":job.get("title") or "Untitled Job","location":job.get("location") or "Not specified",
        "risk_score":result["risk_score"],"probability":result["probability"],"risk_level":result["risk_level"],"prediction":result["prediction"],
    })
    st.session_state.history = st.session_state.history[:50]

# ============================================================
# PDF
# ============================================================

def generate_pdf_report(job, result):
    if not REPORTLAB_AVAILABLE: raise RuntimeError("ReportLab is not installed. Run: pip install reportlab")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36,title="JobGuard Risk Report",author="JobGuard")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("JGTitle",parent=styles["Title"],fontSize=22,leading=27,alignment=TA_CENTER,textColor=colors.HexColor("#16243a"))
    body = ParagraphStyle("JGBody",parent=styles["BodyText"],fontSize=9.5,leading=14,textColor=colors.HexColor("#35445a"))
    heading = ParagraphStyle("JGHeading",parent=styles["Heading2"],fontSize=14,leading=18,textColor=colors.HexColor("#16243a"),spaceBefore=12,spaceAfter=7)
    story=[Paragraph("JobGuard",title_style),Paragraph("AI Job Scam Detection & Explainable Risk Analysis",body),Spacer(1,12)]
    story += [Paragraph(f"<b>Job Title:</b> {html.escape(clean_text(job.get('title')) or 'Not provided')}",body),
              Paragraph(f"<b>Location:</b> {html.escape(clean_text(job.get('location')) or 'Not provided')}",body),
              Paragraph(f"<b>Prediction:</b> {html.escape(result['prediction'])}",body),
              Paragraph(f"<b>Risk Score:</b> {result['risk_score']:.1f}/100",body),
              Paragraph(f"<b>Fraud Probability:</b> {result['probability']*100:.2f}%",body),Spacer(1,12),Paragraph("Observable Risk Signals",heading)]
    data=[["Severity","Risk Signal","Why it matters"]]+[[s["severity"],s["signal"],s["reason"]] for s in result["signals"]]
    table=Table(data,colWidths=[65,145,310],repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16243a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#b9c4d3")),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7fb")])]))
    story.append(table); story += [Spacer(1,12),Paragraph("Safety guidance: independently verify the employer, recruiter identity, official website, email domain and recruitment process. Never pay money or share passwords, OTPs or banking credentials to obtain a job.",body)]
    doc.build(story)
    return buf.getvalue()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="jg-brand"><div class="jg-logo">🛡️</div><div><div class="jg-brand-name">JobGuard</div><div class="jg-brand-sub">AI-powered job scam detection</div></div></div>',unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("Go to",["Analyze Job","Analysis History","About JobGuard"],index=["Analyze Job","Analysis History","About JobGuard"].index(st.session_state.page),label_visibility="collapsed",key="nav_page")
    st.session_state.page = page
    st.markdown("---")
    st.markdown("### Risk Scale")
    st.markdown("🟢 **0–29** — Low Risk")
    st.markdown("🟡 **30–59** — Medium Risk")
    st.markdown("🟠 **60–79** — High Risk")
    st.markdown("🔴 **80–100** — Very High Risk")
    st.markdown("---")
    h = st.session_state.history
    st.markdown("### Session")
    st.metric("Analyses",len(h))
    st.metric("High Risk",sum(x["risk_score"]>=60 for x in h))
    st.caption(f"Average risk: {np.mean([x['risk_score'] for x in h]):.1f}/100" if h else "Average risk: 0.0/100")

# ============================================================
# HEADER / STATUS
# ============================================================

st.markdown('<div class="jg-eyebrow">MACHINE LEARNING · RISK ANALYTICS</div><div class="jg-title">JobGuard</div><div class="jg-subtitle">AI-powered job scam detection and explainable risk analysis</div>',unsafe_allow_html=True)

if model_errors:
    with st.expander("⚠️ Model configuration status",expanded=False):
        for err in model_errors: st.warning(err)
else:
    st.markdown('<div class="success-box">✓ JobGuard ML model and preprocessing pipeline loaded successfully.</div>',unsafe_allow_html=True)

# ============================================================
# ANALYZE PAGE
# ============================================================

if page == "Analyze Job":
    st.markdown('<div class="hero"><div class="jg-eyebrow">PROTECT YOUR NEXT CAREER MOVE</div><h1>Analyze a job before you apply</h1><p>Enter a job manually, load a demonstration, import TXT/CSV data, then run the trained JobGuard pipeline. The dashboard reports fraud probability, a 0–100 risk score, prediction, and observable warning signals.</p></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Start</div><div class="section-sub">Use a demo to validate the workflow, or load your own posting.</div>',unsafe_allow_html=True)
    d1,d2,d3 = st.columns([2.8,1,1])
    with d1:
        selected_demo = st.selectbox("Demonstration posting",list(DEMO_JOBS.keys()),key="demo_selector")
    with d2:
        st.write(""); st.button("Load Demo",type="secondary",use_container_width=True,on_click=load_demo_callback)
    with d3:
        st.write(""); st.button("Clear Form",use_container_width=True,on_click=clear_form_callback)
    if st.session_state.demo_loaded:
        st.markdown(f'<div class="notice">Loaded: <b>{html.escape(selected_demo)}</b>. Review the fields below and click Analyze Job Posting.</div>',unsafe_allow_html=True)

    with st.expander("📥 Import Job Posting",expanded=False):
        st.caption("Supported formats: TXT and CSV. CSV files can contain multiple rows; choose the row to analyze.")
        uploaded = st.file_uploader("Upload TXT or CSV",type=["txt","csv"],key="job_upload")
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".txt"):
                    imported = parse_txt_job(uploaded.read().decode("utf-8",errors="replace"))
                    if imported:
                        st.session_state.imported_data = imported
                        st.success(f"TXT loaded successfully — {uploaded.name}")
                else:
                    df, first_job = parse_csv_file(uploaded)
                    st.success(f"CSV loaded successfully — {len(df):,} row(s) found.")
                    st.dataframe(df.head(10),use_container_width=True,hide_index=True)
                    row_no = st.number_input("CSV row to load",min_value=1,max_value=len(df),value=1,step=1,key="csv_row")
                    if st.button("Load Selected CSV Row",use_container_width=True):
                        st.session_state.imported_data = parse_csv_row(df.iloc[int(row_no)-1].to_dict())
                        st.rerun()
            except Exception as exc:
                st.error(f"Could not import the file: {exc}")
        if st.session_state.imported_data:
            st.info("Imported data is ready. Click the button below to place it into the form.")
            st.button("Load Imported Data",type="primary",use_container_width=True,on_click=load_imported_callback)

    st.markdown('<div class="section-title">Job Information</div><div class="section-sub">Complete fields improve the quality and interpretability of the analysis.</div>',unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        title = st.text_input("Job Title",key="field_title",placeholder="e.g. Software Engineer")
        location = st.text_input("Location",key="field_location",placeholder="e.g. Hyderabad, Telangana, India")
        department = st.text_input("Department",key="field_department",placeholder="e.g. Engineering")
        salary_range = st.text_input("Salary Range",key="field_salary_range",placeholder="e.g. 600000-900000")
        industry = st.text_input("Industry",key="field_industry",placeholder="e.g. Information Technology")
        function = st.text_input("Job Function",key="field_function",placeholder="e.g. Engineering")
    with c2:
        employment_options=["","Full-time","Part-time","Contract","Temporary","Internship","Other"]
        current=st.session_state.get("field_employment_type","")
        employment_type=st.selectbox("Employment Type",employment_options,index=employment_options.index(current) if current in employment_options else 0,key="field_employment_type")
        required_experience=st.text_input("Required Experience",key="field_required_experience",placeholder="e.g. Entry level / Mid-level")
        required_education=st.text_input("Required Education",key="field_required_education",placeholder="e.g. Bachelor's Degree")
        telecommuting=st.checkbox("Remote / Telecommuting",key="field_telecommuting")
        has_company_logo=st.checkbox("Company logo present",key="field_has_company_logo")
        has_questions=st.checkbox("Application questions present",key="field_has_questions")

    company_profile=st.text_area("Company Profile",key="field_company_profile",height=120,placeholder="Company background, products, services and business information...")
    description=st.text_area("Job Description *",key="field_description",height=210,placeholder="Paste the complete job description here...")
    requirements=st.text_area("Requirements",key="field_requirements",height=160,placeholder="Skills, qualifications, experience and responsibilities...")
    benefits=st.text_area("Benefits",key="field_benefits",height=120,placeholder="Insurance, leave, incentives, professional development, etc.")

    st.markdown("")
    analyze = st.button("🔍 Analyze Job Posting",type="primary",use_container_width=True)
    if analyze:
        job={"title":title,"location":location,"department":department,"salary_range":salary_range,"company_profile":company_profile,"description":description,"requirements":requirements,"benefits":benefits,"employment_type":employment_type,"required_experience":required_experience,"required_education":required_education,"industry":industry,"function":function,"telecommuting":int(telecommuting),"has_company_logo":int(has_company_logo),"has_questions":int(has_questions)}
        if not any(clean_text(job[k]) for k in ["title","description","requirements"]):
            st.error("Please enter at least a job title, description, or requirements before analyzing.")
        else:
            with st.spinner("Analyzing with the JobGuard prediction pipeline..."):
                try:
                    result=predict_job(job); save_history(job,result); st.session_state.last_result={"job":job,"result":result}
                    st.success("Analysis completed successfully.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    with st.expander("Technical details"): st.code(traceback.format_exc())

    # ========================================================
    # RESULTS
    # ========================================================
    if st.session_state.last_result:
        job=st.session_state.last_result["job"]; result=st.session_state.last_result["result"]
        level,_,icon=get_risk_level(result["risk_score"])
        risk_color={"Low Risk":"#22c55e","Medium Risk":"#f59e0b","High Risk":"#f97316","Very High Risk":"#ef4444"}[level]
        st.markdown('<div class="section-title">JobGuard Analysis</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="risk-banner" style="border-left:5px solid {risk_color};"><div style="font-size:1.35rem;font-weight:900;">{icon} {html.escape(result["prediction"])} <span style="color:#64748b;">·</span> <span style="color:{risk_color};">{html.escape(level)}</span></div><div style="color:#94a3b8;margin-top:8px;">Estimated fraud probability <b>{result["probability"]*100:.2f}%</b> · Risk score <b>{result["risk_score"]:.1f}/100</b></div></div>',unsafe_allow_html=True)

        m1,m2,m3,m4=st.columns(4)
        for col,label,value,note in [
            (m1,"Risk Score",f"{result['risk_score']:.1f}/100","Overall posting risk"),
            (m2,"Fraud Probability",f"{result['probability']*100:.2f}%","Model-estimated probability"),
            (m3,"Prediction",result["prediction"],"Classification threshold"),
            (m4,"Risk Level",result["risk_level"],"Risk category"),
        ]:
            with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{risk_color if label=="Risk Level" else "#f8fafc"};">{html.escape(str(value))}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

        chart1,chart2=st.columns(2)
        with chart1:
            st.subheader("Risk Score")
            fig=go.Figure(go.Indicator(mode="gauge+number",value=result["risk_score"],number={"suffix":"/100","font":{"size":36,"color":"#f8fafc"}},title={"text":result["risk_level"],"font":{"size":17,"color":"#cbd5e1"}},gauge={"axis":{"range":[0,100],"tickcolor":"#94a3b8"},"bar":{"color":risk_color,"thickness":.25},"bgcolor":"#0f172a","bordercolor":"#334155","steps":[{"range":[0,30],"color":"#123d32"},{"range":[30,60],"color":"#403816"},{"range":[60,80],"color":"#40271d"},{"range":[80,100],"color":"#401d28"}]}))
            fig.update_layout(height=350,margin=dict(l=25,r=25,t=60,b=15),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with chart2:
            st.subheader("Prediction Probability")
            fraud=result["probability"]*100; legit=100-fraud
            fig=go.Figure(go.Pie(labels=["Legitimate","Fraudulent"],values=[legit,fraud],hole=.62,textinfo="label+percent",marker={"colors":["#22c55e","#ef4444"],"line":{"color":"#070b14","width":3}}))
            fig.update_layout(height=350,margin=dict(l=15,r=15,t=35,b=15),paper_bgcolor="rgba(0,0,0,0)",showlegend=True,font={"color":"#cbd5e1"})
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        st.subheader("Observable Risk Signals")
        signal_df=pd.DataFrame(result["signals"])[["severity","signal","reason"]].rename(columns={"severity":"Severity","signal":"Risk Signal","reason":"Why it matters"})
        st.dataframe(signal_df,use_container_width=True,hide_index=True,column_config={"Severity":st.column_config.TextColumn("Severity",width="small"),"Risk Signal":st.column_config.TextColumn("Risk Signal",width="medium"),"Why it matters":st.column_config.TextColumn("Why it matters",width="large")})

        st.subheader("Risk Indicator Profile")
        combined=" ".join(clean_text(job.get(k)) for k in ["title","description","requirements","benefits","company_profile"]).lower()
        profile={
            "Payment Risk":int(any(x in combined for x in ["registration fee","processing fee","pay upfront","deposit","payment required"])),
            "Financial Data":int(any(x in combined for x in ["bank account","bank details","credit card","debit card","financial information"])),
            "Urgency":int(any(x in combined for x in ["urgent","immediately","act now","hurry","asap"])),
            "Guaranteed Income":int(any(x in combined for x in ["guaranteed income","guaranteed salary","guaranteed earnings"])),
            "Missing Company Info":int(not clean_text(job.get("company_profile"))),
            "Short Description":int(len(clean_text(job.get("description")))<120),
            "Missing Salary":int(not clean_text(job.get("salary_range"))),
        }
        pdf=pd.DataFrame({"Indicator":list(profile.keys()),"Detected":list(profile.values())})
        fig=px.bar(pdf,x="Detected",y="Indicator",orientation="h",range_x=[0,1],text="Detected")
        fig.update_traces(marker_color="#6366f1",textposition="outside")
        fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#cbd5e1"},xaxis={"tickvals":[0,1],"ticktext":["No","Yes"],"gridcolor":"#1e293b"},yaxis={"title":""},margin=dict(l=20,r=45,t=20,b=25))
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        if result["source"] == "Trained ML model":
            st.markdown('<div class="success-box">✓ Prediction generated using the trained JobGuard machine-learning model.</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">⚠ The trained preprocessing pipeline could not transform this input, so the transparent fallback risk rules were used. This result should be treated as a fallback assessment.</div>',unsafe_allow_html=True)
            if result["model_error"]:
                with st.expander("Model transformation diagnostic"): st.code(result["model_error"])

        st.subheader("Job Summary")
        s1,s2=st.columns(2)
        with s1:
            st.write(f"**Job Title:** {job.get('title') or 'Not provided'}")
            st.write(f"**Location:** {job.get('location') or 'Not provided'}")
            st.write(f"**Department:** {job.get('department') or 'Not provided'}")
            st.write(f"**Employment Type:** {job.get('employment_type') or 'Not provided'}")
        with s2:
            st.write(f"**Industry:** {job.get('industry') or 'Not provided'}")
            st.write(f"**Function:** {job.get('function') or 'Not provided'}")
            st.write(f"**Experience:** {job.get('required_experience') or 'Not provided'}")
            st.write(f"**Salary:** {job.get('salary_range') or 'Not provided'}")

        st.subheader("Safety Recommendations")
        recs=["Verify the company's official website.","Check the recruiter's company email domain.","Never pay money to obtain a job.","Never share passwords or OTPs.","Never provide banking credentials.","Verify unusually high salary claims.","Research the recruiter independently.","Be cautious of urgent hiring pressure."]
        r1,r2=st.columns(2)
        for i,rec in enumerate(recs):
            with (r1 if i%2==0 else r2): st.write(f"✓ {rec}")

        st.subheader("Analysis Report")
        try:
            pdf_bytes=generate_pdf_report(job,result)
            safe=re.sub(r"[^A-Za-z0-9]+","_",job.get("title") or "Analysis").strip("_")
            st.download_button("📄 Download PDF Analysis Report",data=pdf_bytes,file_name=f"JobGuard_{safe or 'Analysis'}_Report.pdf",mime="application/pdf",use_container_width=True)
        except Exception as exc:
            st.warning(f"PDF report unavailable: {exc}")

# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "Analysis History":
    st.markdown('<div class="section-title">Analysis History</div><div class="section-sub">Records from the current Streamlit browser session.</div>',unsafe_allow_html=True)
    history=st.session_state.history
    if not history:
        st.info("No analyses yet. Go to Analyze Job and run an analysis.")
    else:
        df=pd.DataFrame(history)
        display=df.rename(columns={"time":"Time","title":"Job Title","location":"Location","risk_score":"Risk Score","probability":"Fraud Probability","risk_level":"Risk Level","prediction":"Prediction"}).copy()
        display["Risk Score"]=display["Risk Score"].round(1)
        display["Fraud Probability"]=(display["Fraud Probability"]*100).round(2).astype(str)+"%"
        st.dataframe(display,use_container_width=True,hide_index=True)
        a,b,c,d=st.columns(4)
        a.metric("Total Analyses",len(history)); b.metric("High Risk",sum(x["risk_score"]>=60 for x in history)); c.metric("Fraudulent",sum(x["prediction"]=="Fraudulent" for x in history)); d.metric("Average Risk",f"{np.mean([x['risk_score'] for x in history]):.1f}")
        chart=pd.DataFrame(history)
        fig=px.bar(chart,x="time",y="risk_score",hover_name="title",labels={"time":"Analysis Time","risk_score":"Risk Score"})
        for y,label in [(30,"Medium Risk"),(60,"High Risk"),(80,"Very High Risk")]: fig.add_hline(y=y,line_dash="dash",annotation_text=label)
        fig.update_layout(height=400,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"color":"#cbd5e1"},xaxis={"gridcolor":"#1e293b"},yaxis={"range":[0,100],"gridcolor":"#1e293b"})
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        e1,e2=st.columns(2)
        with e1:
            st.download_button("Download History CSV",data=pd.DataFrame(history).to_csv(index=False).encode("utf-8"),file_name="JobGuard_Analysis_History.csv",mime="text/csv",use_container_width=True)
        with e2:
            if st.button("Clear Analysis History",use_container_width=True):
                st.session_state.history=[]; st.session_state.last_result=None; st.rerun()

# ============================================================
# ABOUT PAGE
# ============================================================

else:
    st.markdown('<div class="section-title">About JobGuard</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="jg-eyebrow">JOBGUARD</div><h2>AI-powered job scam detection</h2><p style="color:#94a3b8;line-height:1.8;">JobGuard is a machine-learning-based decision-support application that evaluates job postings and combines a trained classification pipeline with transparent observable risk signals.</p></div>',unsafe_allow_html=True)
    a1,a2=st.columns(2)
    with a1:
        st.subheader("How it works")
        st.markdown("**1. Job Input** — Manual fields, TXT import, CSV import, or demonstrations.\n\n**2. Feature Engineering** — Text length, missing information, salary information and observable scam indicators are derived.\n\n**3. Machine Learning** — The saved preprocessing pipeline transforms the engineered row for the trained model.\n\n**4. Risk Probability** — The model estimates the probability that the posting resembles fraudulent examples.\n\n**5. Risk Score** — Probability is represented as a 0–100 score.\n\n**6. Explainability** — Observable warning signals are shown alongside the model result.")
    with a2:
        st.subheader("Risk scale")
        st.markdown("🟢 **0–29 — Low Risk**\n\n🟡 **30–59 — Medium Risk**\n\n🟠 **60–79 — High Risk**\n\n🔴 **80–100 — Very High Risk**")
    st.subheader("Model Information")
    metadata_df=pd.DataFrame([
        {"Property":"Model","Value":metadata.get("model_name","Unknown") if isinstance(metadata,dict) else "Unknown"},
        {"Property":"Decision Threshold","Value":metadata.get("threshold","Unknown") if isinstance(metadata,dict) else "Unknown"},
        {"Property":"ROC-AUC","Value":f"{metadata.get('roc_auc'):.4f}" if isinstance(metadata,dict) and isinstance(metadata.get("roc_auc"),(int,float)) else "Unknown"},
        {"Property":"PR-AUC","Value":f"{metadata.get('pr_auc'):.4f}" if isinstance(metadata,dict) and isinstance(metadata.get("pr_auc"),(int,float)) else "Unknown"},
        {"Property":"Feature Count","Value":metadata.get("feature_count","Unknown") if isinstance(metadata,dict) else "Unknown"},
    ])
    st.dataframe(metadata_df,use_container_width=True,hide_index=True)
    st.markdown('<div class="warning-box"><b>Important:</b> JobGuard is a decision-support tool. A high-risk prediction does not prove that a job is fraudulent, and a low-risk prediction does not guarantee legitimacy. Always independently verify employers and recruiters.</div>',unsafe_allow_html=True)

st.markdown("---")
st.caption("JobGuard • AI Job Scam Detection & Risk Analytics • Machine Learning + Explainable Risk Signals")
