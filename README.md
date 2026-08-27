# JobGuard — Job Scam Detection & Risk Analytics Platform

JobGuard is a machine-learning-powered job scam detection and risk analytics platform that evaluates job postings and estimates their likelihood of being fraudulent.

The project combines data analysis, feature engineering, machine learning, risk scoring, explainable warning indicators, interactive visualization, and Streamlit deployment into an end-to-end analytics solution.

## Live Demo

**[Launch JobGuard](https://jobguard-z7b46dagpe9kb3prqhf8cv.streamlit.app/)**

## GitHub Repository

**[View Source Code](https://github.com/Kodhanda1223/JobGuard)**

---

## Project Overview

Online job scams can contain observable warning signals such as:

- Requests for money or payment
- Requests for bank or financial information
- Urgent action requirements
- Guaranteed income claims
- Suspicious job descriptions
- Unusual posting characteristics

JobGuard analyzes these signals along with textual and structured job-posting features and produces:

- Fraud probability
- 0–100 risk score
- Risk classification
- Observable warning signals
- Analysis history
- Interactive analytics
- Downloadable reports

---

## Objectives

The main objectives of JobGuard are to:

1. Detect potentially fraudulent job postings.
2. Quantify job-posting risk using a 0–100 score.
3. Identify observable warning signals.
4. Apply machine learning to textual and structured job data.
5. Build an interactive analytics dashboard.
6. Make model predictions easier to interpret.
7. Demonstrate an end-to-end Data Analytics and Machine Learning workflow.

---

## Machine Learning Approach

JobGuard uses a Logistic Regression classification model.

### Machine Learning Workflow

```text
Raw Job Dataset
       |
       v
Data Understanding
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Text + Structured Features
       |
       v
Preprocessing Pipeline
       |
       v
Logistic Regression
       |
       v
Fraud Probability
       |
       v
Risk Score
       |
       v
Risk Classification
       |
       v
Warning Signals
       |
       v
Streamlit Dashboard