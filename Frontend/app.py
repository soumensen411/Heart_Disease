import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import heart_model  # noqa: F401  (needed so joblib can unpickle the pipeline)
from heart_model import INPUT_COLUMNS

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .no-disease {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .disease {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: rgba(227, 242, 253, 0.15);
        color: inherit;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .card-box {
        background-color: rgba(128, 128, 128, 0.08);
        color: inherit;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin: 10px 0;
    }
    .warning-box {
        background-color: rgba(255, 243, 224, 0.15);
        color: inherit;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    .danger-box {
        background-color: rgba(255, 235, 238, 0.15);
        color: inherit;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        background-color: #e74c3c;
        color: white;
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved pipeline (cleaning + scaling + one-hot encoding + KNN in one object)
@st.cache_resource
def load_artifacts():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (here, '.', '/app', '/mount/src', os.path.join(here, '..')):
        candidate = os.path.join(path, 'heart_model.pkl')
        if os.path.exists(candidate):
            try:
                return joblib.load(candidate)
            except Exception as e:
                st.error(f"Error loading model file: {e}")
                return None
    st.error("❌ Model file not found! Expected heart_model.pkl next to app.py.")
    st.info("Create it with:  python train.py")
    return None

model = load_artifacts()

# Sidebar navigation
st.sidebar.title("❤️ Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "🔮 Predict", "📊 Batch Prediction"])

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Heart Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered risk assessment using Machine Learning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="card-box" style="text-align: center;">
                <h3 style="color: inherit;">🎯 Accurate</h3>
                <p>Trained on comprehensive heart disease dataset with multiple features</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="card-box" style="text-align: center;">
                <h3 style="color: inherit;">⚡ Fast</h3>
                <p>Get instant predictions with our optimized KNN model</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="card-box" style="text-align: center;">
                <h3 style="color: inherit;">🔒 Secure</h3>
                <p>All predictions are processed locally on your machine</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div class="info-box">
            <h4 style="color: inherit;">📋 How to Use</h4>
            <ol>
                <li>Navigate to the <b>Predict</b> page from the sidebar</li>
                <li>Enter your health parameters in the form</li>
                <li>Click <b>Predict</b> to get your heart disease risk assessment</li>
                <li>For multiple patients, use the <b>Batch Prediction</b> feature</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("❌ Model files not found. Please check your PKL files.")

# Predict Page
elif page == "🔮 Predict":
    st.markdown('<div class="main-header">Single Patient Prediction</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please check your PKL files.")
        st.stop()

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Demographics & Vitals")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["M", "F"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.subheader("🏥 Clinical Features")
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # Create input dataframe
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([input_data])[INPUT_COLUMNS]

    st.markdown("---")

    if st.button("🔮 Predict Heart Disease Risk"):
        with st.spinner("Analyzing..."):
            # The pipeline cleans, encodes and scales the raw row, then predicts
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            # Display results
            st.markdown("---")
            st.subheader("📋 Prediction Results")

            if prediction == 0:
                st.markdown(
                    f'<div class="prediction-box no-disease">✅ No Heart Disease Detected</div>',
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    f'<div class="prediction-box disease">⚠️ Heart Disease Risk Detected</div>',
                    unsafe_allow_html=True
                )

            # Confidence display
            confidence = max(prediction_proba) * 100
            st.progress(int(confidence))
            st.info(f"Model Confidence: **{confidence:.2f}%**")

            # What the KNN actually sees for this patient
            st.subheader("📊 Feature Analysis")

            fig, ax = plt.subplots(figsize=(10, 6))
            feature_values = model[:-1].transform(input_df)[0]
            feature_names = model.named_steps['prep'].get_feature_names_out()
            feature_names = [n.split('__', 1)[-1] for n in feature_names]

            colors = ['#e74c3c' if v > 0 else '#3498db' for v in feature_values]
            bars = ax.barh(range(len(feature_names)), feature_values, color=colors)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=10)
            ax.set_xlabel('Value as seen by the model (numbers are standardised)')
            ax.set_title('Model Input Features')
            plt.tight_layout()
            st.pyplot(fig)

# Batch Prediction Page
elif page == "📊 Batch Prediction":
    st.markdown('<div class="main-header">Batch Prediction</div>', unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please check your PKL files.")
        st.stop()

    st.markdown("""
        <div class="info-box">
            <p>Upload a CSV file with patient data. The file should contain columns:</p>
            <p><code>Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope</code></p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} records")

            with st.expander("📋 Preview Data"):
                st.dataframe(batch_df.head())

            if st.button("🔮 Run Batch Prediction"):
                with st.spinner("Processing all records..."):
                    missing = [c for c in INPUT_COLUMNS if c not in batch_df.columns]
                    if missing:
                        raise ValueError(f"Missing columns: {', '.join(missing)}")
                    batch_in = batch_df[INPUT_COLUMNS]
                    predictions = model.predict(batch_in)
                    probabilities = model.predict_proba(batch_in)

                    # Add results to dataframe
                    results_df = batch_df.copy()
                    results_df['Prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in predictions]
                    results_df['Confidence'] = [max(prob) * 100 for prob in probabilities]
                    results_df['Risk_Score'] = probabilities[:, 1] * 100

                    st.markdown("---")
                    st.subheader("📊 Results Summary")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patients", len(results_df))
                    with col2:
                        at_risk = sum(predictions)
                        st.metric("At Risk", int(at_risk), delta=f"{at_risk/len(predictions)*100:.1f}%")
                    with col3:
                        healthy = len(predictions) - at_risk
                        st.metric("Healthy", int(healthy), delta=f"{healthy/len(predictions)*100:.1f}%")

                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                    # Pie chart
                    labels = ['Healthy', 'At Risk']
                    sizes = [healthy, at_risk]
                    colors = ['#2ecc71', '#e74c3c']
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Prediction Distribution')

                    # Risk score distribution
                    ax2.hist(results_df['Risk_Score'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
                    ax2.set_xlabel('Risk Score (%)')
                    ax2.set_ylabel('Number of Patients')
                    ax2.set_title('Risk Score Distribution')
                    ax2.axvline(x=50, color='red', linestyle='--', label='Threshold')
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='heart_disease_predictions.csv',
                        mime='text/csv'
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV has the correct column names and format.")