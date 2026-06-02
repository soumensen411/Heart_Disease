import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

# Load models and artifacts - checks multiple paths for deployment compatibility
@st.cache_resource
def load_artifacts():
    # List of possible paths to check (in order of priority)
    possible_paths = [
        '.',                                    # Current directory (local run)
        os.path.dirname(os.path.abspath(__file__)),  # Script directory
        '/app',                                 # Common Docker/Cloud path
        '/mount/src',                           # Streamlit Cloud path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),  # Parent dir
    ]

    # Also check if files are in the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths.append(script_dir)

    # Find the first path that contains all three files
    found_path = None
    for path in possible_paths:
        if all(os.path.exists(os.path.join(path, f)) for f in ['KNN_HeartPKL.pkl', 'scaler.pkl', 'columns.pkl']):
            found_path = path
            break

    # If not found, search recursively in script directory (up to 2 levels)
    if found_path is None:
        for root, dirs, files in os.walk(script_dir):
            depth = root.replace(script_dir, '').count(os.sep)
            if depth > 2:
                continue
            if all(f in files for f in ['KNN_HeartPKL.pkl', 'scaler.pkl', 'columns.pkl']):
                found_path = root
                break

    if found_path is None:
        st.error("❌ Model files not found!")
        st.info("Searched in these locations:")
        for path in possible_paths:
            st.code(path)
        st.info("Please ensure these files are in your project folder:")
        st.code("KNN_HeartPKL.pkl, scaler.pkl, columns.pkl")
        return None, None, None

    try:
        model = joblib.load(os.path.join(found_path, 'KNN_HeartPKL.pkl'))
        scaler = joblib.load(os.path.join(found_path, 'scaler.pkl'))
        columns = joblib.load(os.path.join(found_path, 'columns.pkl'))
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, columns = load_artifacts()

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

    input_df = pd.DataFrame([input_data])

    # One-hot encoding to match training format
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure all columns from training are present
    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training
    input_encoded = input_encoded[columns]

    # Convert to int (as done in training)
    input_encoded = input_encoded.astype(int)

    st.markdown("---")

    if st.button("🔮 Predict Heart Disease Risk"):
        with st.spinner("Analyzing..."):
            # Scale the input
            input_scaled = scaler.transform(input_encoded)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

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

            # Feature importance visualization (using input values as proxy)
            st.subheader("📊 Feature Analysis")

            fig, ax = plt.subplots(figsize=(10, 6))
            feature_values = input_encoded.iloc[0].values
            feature_names = input_encoded.columns

            colors = ['#e74c3c' if v > 0 else '#3498db' for v in feature_values]
            bars = ax.barh(range(len(feature_names)), feature_values, color=colors)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=10)
            ax.set_xlabel('Value')
            ax.set_title('Input Feature Values')
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
                    # One-hot encoding
                    batch_encoded = pd.get_dummies(batch_df, drop_first=True)

                    # Ensure all columns from training are present
                    for col in columns:
                        if col not in batch_encoded.columns:
                            batch_encoded[col] = 0

                    # Reorder columns
                    batch_encoded = batch_encoded[columns]
                    batch_encoded = batch_encoded.astype(int)

                    # Scale and predict
                    batch_scaled = scaler.transform(batch_encoded)
                    predictions = model.predict(batch_scaled)
                    probabilities = model.predict_proba(batch_scaled)

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
