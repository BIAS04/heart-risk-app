import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Heart Risk Analyzer",
    page_icon="⚜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LUXURY THEME CSS ---
# This CSS block creates a simple, elegant dark theme with gold accents.
st.markdown("""
<style>
    /* Base text and background */
    body {
        color: #E0E0E0;
    }
    .stApp {
        background-color: #1E1E1E;
    }

    /* Title */
    h1 {
        color: #D4AF37; /* Muted Gold */
        font-family: 'Garamond', serif;
        text-align: center;
        font-weight: 400;
    }

    /* Subheaders */
    h2, h3 {
        color: #E0E0E0;
        font-family: 'Garamond', serif;
        font-weight: 400;
    }
    
    /* Input Form Container */
    .st-emotion-cache-z5fcl4 {
        background-color: #2a2a2e;
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 25px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2a2a2e;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border: 2px solid #D4AF37;
        border-radius: 8px;
        color: #D4AF37;
        background-color: transparent;
        transition: all 0.3s ease;
        padding: 10px 0;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton > button:hover, .stButton > button:focus {
        background-color: #D4AF37;
        color: #1E1E1E;
        border-color: #D4AF37;
    }

    /* Alert Boxes - Simple Style */
    div[data-baseweb="alert"] {
        background-color: transparent;
        border-radius: 8px;
        border: 1px solid;
        padding: 1rem;
        text-align: center;
    }
    .st-alert-error {
        border-color: #ff4b4b; /* Red */
    }
    .st-alert-success {
        border-color: #26A86B; /* Green */
    }

</style>
""", unsafe_allow_html=True)


# --- LOAD ASSETS ---
@st.cache_data
def load_assets():
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, expected_columns = load_assets()

if not all([model, scaler, expected_columns]):
    st.error("Error: Critical model files are missing.")
    st.stop()


# --- SIDEBAR ---
with st.sidebar:
    st.title("⚜️ Heart Risk Analyzer")
    st.write("This machine learning application provides a predictive analysis of heart disease risk based on key health metrics.")
    st.info("This is a demonstration and not a substitute for professional medical advice.")


# --- HEADER ---
st.title("Heart Disease Risk Analyzer")
st.markdown("Enter patient data for a predictive risk assessment.")


# --- INPUT FORM ---
with st.container():
    st.subheader("Patient Vitals & Health Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

    with col2:
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

    with col3:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])

    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# --- PREDICTION LOGIC ---
if st.button("Analyze Risk"):
    with st.spinner("Analyzing..."):
        time.sleep(0.5) # Short delay

        raw_input = {col: 0 for col in expected_columns}
        raw_input.update({
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
            'MaxHR': max_hr, 'Oldpeak': oldpeak, 'Sex_' + sex: 1, 'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1, 'ExerciseAngina_' + exercise_angina: 1, 'ST_Slope_' + st_slope: 1
        })
        
        input_df = pd.DataFrame([raw_input])[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # --- DISPLAY RESULTS ---
        st.write("---")
        st.subheader("Analysis Complete")
        
        result_col1, result_col2 = st.columns([1, 1.5])
        with result_col1:
            if prediction == 1:
                st.error("Finding: High Risk Detected")
            else:
                st.success("Finding: Low Risk Detected")
            st.markdown(f"The model calculated a **{probability:.2%}** probability of heart disease.")
            st.caption("Disclaimer: This is a predictive model, not a medical diagnosis.")
        
        with result_col2:
            # Create gauge chart with the luxury theme
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': "%", 'font': {'size': 50, 'color': '#E0E0E0'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score", 'font': {'size': 24, 'color': '#D4AF37'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#E0E0E0"},
                    'bar': {'color': '#D4AF37'},
                    'bgcolor': "#1E1E1E",
                    'borderwidth': 2,
                    'bordercolor': "#4a4a4a",
                    'steps': [{'range': [0, 50], 'color': 'rgba(42, 110, 69, 0.5)'}, # Muted green
                              {'range': [50, 100], 'color': 'rgba(110, 42, 42, 0.5)'}]  # Muted red
                }))
            fig.update_layout(
                paper_bgcolor="#2a2a2e",
                font={'color': '#E0E0E0', 'family': "Arial"},
                height=250, margin={'t':0, 'b':0, 'l':20, 'r':20}
            )
            st.plotly_chart(fig, use_container_width=True)