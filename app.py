import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="TalentRisk Premier | Interactive HR Analyst",
    page_icon="🧬",
    layout="wide"
)

# Advanced Modern Executive Styling with Premium Colors
st.markdown("""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap");

    /* Global font & background upgrade */
    * {
        font-family: "Plus Jakarta Sans", sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    /* Sophisticated Header Banner with Vibrant Color Gradient */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 40px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.15);
        border-left: 8px solid #6366f1;
    }
    .header-title {
        font-weight: 800;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -1px;
    }
    .header-subtitle {
        font-size: 1.15rem;
        opacity: 0.90;
        margin-top: 10px;
        font-weight: 400;
    }

    /* Extreme High-Visibility Contrast Labels (Styled & Cleanly Spaced) */
    label, .stWidgetLabel, div[data-testid="stWidgetLabel"] p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #0f172a !important; /* Near black Slate-900 */
        letter-spacing: 0.2px;
        margin-bottom: 6px !important;
    }

    /* Beautiful custom section header labels with neon borders */
    .section-banner-blue {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #1e3a8a !important;
        border-left: 5px solid #2563eb !important;
        padding-left: 12px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-banner-indigo {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #312e81 !important;
        border-left: 5px solid #4f46e5 !important;
        padding-left: 12px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-banner-violet {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #581c87 !important;
        border-left: 5px solid #7c3aed !important;
        padding-left: 12px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom subheaders for analysis outputs */
    .custom-subheader {
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        color: #0f172a !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
        border-left: 8px solid #4f46e5 !important;
        padding-left: 16px !important;
    }

    /* Highly polished card form containers */
    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 35px !important;
        border-radius: 16px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05) !important;
    }

    /* High impact Metric formatting */
    div[data-testid="stMetricValue"] {
        font-size: 3rem !important;
        font-weight: 900 !important;
        color: #090d16 !important; /* Pitch black */
    }
    div[data-testid="stMetricDelta"] p {
        font-size: 1.3rem !important;
        font-weight: 900 !important;
    }

    /* Premium Custom Action Button */
    div.stButton > button {
        background: linear-gradient(90deg, #312e81 0%, #4338ca 100%) !important;
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        width: 100% !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(67, 56, 202, 0.25) !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1e1b4b 0%, #312e81 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(67, 56, 202, 0.35) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. LOAD SAVED MODEL
# --------------------------------------------------
model_data = joblib.load('/content/employee_attrition_model.pkl')

model = model_data['model']
preprocessor = model_data['preprocessor']
label_encoder = model_data['label_encoder']

feature_columns = model_data['feature_columns']
categorical_features = model_data['categorical_features']
numerical_features = model_data['numerical_features']

category_options = model_data['category_options']
numeric_defaults = model_data['numeric_defaults']

# --------------------------------------------------
# Helper to dynamically inject formatting spaces
# --------------------------------------------------
def clean_label(text):
    # Turn camelCase or PascalCase into spaced words
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    # Fix abbreviations
    spaced = spaced.replace("Curr ", "Current ").replace("Rate", " Rate").replace("  ", " ")
    return spaced.strip()

# --------------------------------------------------
# 3. NUMERICAL INPUT LIMITS
# --------------------------------------------------
numeric_limits = {
    'Age': (18, 60),
    'DailyRate': (102, 1499),
    'DistanceFromHome': (1, 29),
    'Education': (1, 5),
    'EnvironmentSatisfaction': (1, 4),
    'HourlyRate': (30, 100),
    'JobInvolvement': (1, 4),
    'JobLevel': (1, 5),
    'JobSatisfaction': (1, 4),
    'MonthlyIncome': (1009, 19999),
    'MonthlyRate': (2094, 26999),
    'NumCompaniesWorked': (0, 9),
    'PercentSalaryHike': (11, 25),
    'PerformanceRating': (3, 4),
    'RelationshipSatisfaction': (1, 4),
    'StockOptionLevel': (0, 3),
    'TotalWorkingYears': (0, 40),
    'TrainingTimesLastYear': (0, 6),
    'WorkLifeBalance': (1, 4),
    'YearsAtCompany': (0, 40),
    'YearsInCurrentRole': (0, 18),
    'YearsSinceLastPromotion': (0, 15),
    'YearsWithCurrManager': (0, 17)
}

def run_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    processed_data = preprocessor.transform(input_df)
    prediction = model.predict(processed_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(processed_data)[0]
    yes_index = list(model.classes_).index(label_encoder.transform(["Yes"])[0])
    leave_probability = probabilities[yes_index] * 100
    return predicted_label, leave_probability

# --------------------------------------------------
# 4. APP TITLE LAYOUT
# --------------------------------------------------
st.markdown("""
    <div class="header-container">
        <div class="header-title">🧬 TalentRisk AI Dashboard</div>
        <div class="header-subtitle">An executive intelligence workspace engineered to decode personnel attrition markers and simulate strategic retention models.</div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Employee Profiler Workspace", "⚡ Interactive Retention Simulator"])

if 'base_inputs' not in st.session_state:
    st.session_state.base_inputs = {}

# --------------------------------------------------
# TAB 1: EMPLOYEE PROFILER WORKSPACE
# --------------------------------------------------
with tab1:
    st.markdown("<p style='font-size: 1.15rem; font-weight: 500; color: #475569; margin-bottom: 20px;'>Complete the analytical profile form below to compute the initial attrition risk baseline.</p>", unsafe_allow_html=True)

    with st.form("employee_form", clear_on_submit=False):
        form_inputs = {}

        st.markdown('<div class="section-banner-blue">🏢 Department & Organizational Role Profile</div>', unsafe_allow_html=True)
        col_a1, col_a2, col_a3 = st.columns(3)
        role_features = ['Department', 'JobRole', 'JobLevel', 'JobInvolvement', 'BusinessTravel', 'OverTime']
        for i, feat in enumerate(role_features):
            label_display = clean_label(feat)
            with (col_a1 if i % 3 == 0 else col_a2 if i % 3 == 1 else col_a3):
                if feat in categorical_features:
                    form_inputs[feat] = st.selectbox(label_display, category_options[feat], index=0)
                else:
                    min_v, max_v = numeric_limits[feat]
                    form_inputs[feat] = st.number_input(label_display, min_v, max_v, int(numeric_defaults[feat]))

        st.markdown('<div class="section-banner-indigo">💸 Compensation Structure & Demographics</div>', unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns(3)
        comp_features = ['MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel', 'Age', 'Gender', 'MaritalStatus']
        for i, feat in enumerate(comp_features):
            label_display = clean_label(feat)
            with (col_b1 if i % 3 == 0 else col_b2 if i % 3 == 1 else col_b3):
                if feat in categorical_features:
                    form_inputs[feat] = st.selectbox(label_display, category_options[feat], index=0)
                else:
                    min_v, max_v = numeric_limits[feat]
                    form_inputs[feat] = st.number_input(label_display, min_v, max_v, int(numeric_defaults[feat]))

        st.markdown('<div class="section-banner-violet">🌱 Historic Tenure & Work Sentiment Factors</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        remaining_features = [f for f in feature_columns if f not in role_features + comp_features]
        for i, feat in enumerate(remaining_features):
            label_display = clean_label(feat)
            with (col_c1 if i % 3 == 0 else col_c2 if i % 3 == 1 else col_c3):
                if feat in categorical_features:
                    form_inputs[feat] = st.selectbox(label_display, category_options[feat], index=0)
                else:
                    min_v, max_v = numeric_limits[feat]
                    form_inputs[feat] = st.number_input(label_display, min_v, max_v, int(numeric_defaults[feat]))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Calculate Executive Risk Check")

    if submitted:
        st.session_state.base_inputs = form_inputs

    if st.session_state.base_inputs:
        base_label, base_prob = run_prediction(st.session_state.base_inputs)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="custom-subheader">Baseline Assessment Result</div>', unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1.2, 2])

        with col_res1:
            if base_label == "Yes":
                st.error("⚠️ Attrition Risk Flagged: High exit indicators present.")
            else:
                st.success("✅ Profile Stable: This employee is highly likely to stay.")

        with col_res2:
            st.metric("Baseline Retention Risk", f" {base_prob:.1f}%")
            st.progress(float(base_prob / 100))

# --------------------------------------------------
# TAB 2: RETENTION MITIGATION SIMULATOR
# --------------------------------------------------
with tab2:
    st.markdown('<div class="custom-subheader">Mitigation Adjuster Workspace</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; font-weight: 600; color: #334155;'>Simulate key strategic alterations to instantly calculate risk impact comparisons.</p>", unsafe_allow_html=True)

    if not st.session_state.base_inputs:
        st.info("💡 To start simulation planning, please configure a baseline profile first inside 'Baseline Profiler Workspace'.")
    else:
        scenario_inputs = st.session_state.base_inputs.copy()
        baseline_label, baseline_prob = run_prediction(st.session_state.base_inputs)

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

        with col_ctrl1:
            st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #1e3a8a; margin-bottom: 12px;'>💳 Compensation Levers</div>", unsafe_allow_html=True)
            scenario_inputs['MonthlyIncome'] = st.slider(
                "Modify Monthly Income ($)",
                min_value=numeric_limits['MonthlyIncome'][0],
                max_value=numeric_limits['MonthlyIncome'][1],
                value=int(scenario_inputs['MonthlyIncome']),
                step=100
            )
            scenario_inputs['PercentSalaryHike'] = st.slider(
                "Modify Salary Hike Percentage (%)",
                min_value=numeric_limits['PercentSalaryHike'][0],
                max_value=numeric_limits['PercentSalaryHike'][1],
                value=int(scenario_inputs['PercentSalaryHike'])
            )

        with col_ctrl2:
            st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #1e3a8a; margin-bottom: 12px;'>⚖️ Schedule & Work Balance Levers</div>", unsafe_allow_html=True)
            ot_index = category_options['OverTime'].index(scenario_inputs['OverTime'])
            scenario_inputs['OverTime'] = st.selectbox(
                "Overtime Policy Status",
                category_options['OverTime'],
                index=ot_index
            )
            scenario_inputs['WorkLifeBalance'] = st.slider(
                "Adjust Work Life Balance Index",
                min_value=numeric_limits['WorkLifeBalance'][0],
                max_value=numeric_limits['WorkLifeBalance'][1],
                value=int(scenario_inputs['WorkLifeBalance'])
            )

        with col_ctrl3:
            st.markdown("<div style='font-size: 1.25rem; font-weight: 800; color: #1e3a8a; margin-bottom: 12px;'>🌱 Workplace Satisfaction Levers</div>", unsafe_allow_html=True)
            scenario_inputs['JobSatisfaction'] = st.slider(
                "Adjust Job Satisfaction Index",
                min_value=numeric_limits['JobSatisfaction'][0],
                max_value=numeric_limits['JobSatisfaction'][1],
                value=int(scenario_inputs['JobSatisfaction'])
            )
            scenario_inputs['EnvironmentSatisfaction'] = st.slider(
                "Adjust Environment Satisfaction Index",
                min_value=numeric_limits['EnvironmentSatisfaction'][0],
                max_value=numeric_limits['EnvironmentSatisfaction'][1],
                value=int(scenario_inputs['EnvironmentSatisfaction'])
            )

        # Re-predict
        new_label, new_prob = run_prediction(scenario_inputs)
        delta_val = new_prob - baseline_prob

        st.markdown("--- ")
        st.markdown('<div class="custom-subheader">Live Scenario Performance Evaluation</div>', unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Baseline Risk Index", value=f"{baseline_prob:.1f}%")
        with col_m2:
            st.metric(
                label="Simulated Risk Index",
                value=f"{new_prob:.1f}%",
                delta=f"{delta_val:+.1f}%",
                delta_color="inverse"
            )
        with col_m3:
            st.write("**Simulation Strategy Outcome**")
            if new_label == "Yes":
                st.error("Action Recommended")
            else:
                st.success("Retention Plan Achieved!")

        st.progress(float(new_prob / 100))

        if new_prob < baseline_prob:
            st.success(f"Optimal Strategy Formulated! The customized changes successfully reduced overall risk probability by **{abs(delta_val):.1f}%**.")
        elif new_prob > baseline_prob:
            st.warning("Mitigation warning: This setup results in an unfavorable increase in predicted attrition risk.")
