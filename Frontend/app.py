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

# Load models and artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('KNN_HeartPKL.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please ensure the following files are in the same directory:")
        st.info("- KNN_HeartPKL.pkl")
        st.info("- scaler.pkl")
        st.info("- columns.pkl")
        return None, None, None

model, scaler, columns = load_artifacts()

# Sidebar navigation
st.sidebar.title("❤️ Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "🔮 Predict", "📊 Batch Prediction", "🇧🇩 বাংলাদেশ কনটেক্সট", "ℹ️ About"])

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

    if model is not None:
        st.success("✅ Model loaded successfully! Ready for predictions.")
    else:
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

                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df)

                    # Visualization
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

                    # Download results
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

# Bengali / Bangladesh Context Page
elif page == "🇧🇩 বাংলাদেশ কনটেক্সট":
    st.markdown('<div class="main-header">বাংলাদেশের প্রেক্ষাপটে হৃদরোগ ঝুঁকি</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Heart Disease Risk in Bangladesh Context</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <h4 style="color: inherit;">⚠️ বাংলাদেশে হৃদরোগের প্রধান ঝুঁকি ফ্যাক্টর (Major Risk Factors in Bangladesh)</h4>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            ### 🍚 খাদ্যাভাস (Dietary Habits)
            - **ভাত বেশি খাওয়া** → High carbohydrate intake
            - **তেল-চর্বি যুক্ত খাবার** → Fried foods (পুরি, পরোটা, সমুচা)
            - **নুন বেশি খাওয়া** → High salt intake (pickles, bharta)
            - **সবজি-ফল কম খাওয়া** → Low fiber intake
            - **গরুর মাংস বেশি, মাছ কম** → Red meat preference

            ### 🚬 জীবনযাত্রা (Lifestyle)
            - **ধূমপান** → Cigarette, bidi (বিড়ি)
            - **নিষ্ক্রিয় জীবনযাত্রা** → Desk jobs, no exercise
            - **মানসিক চাপ** → Traffic, work pressure, financial stress
            - **ঘুমের অভাব** → Late nights, early mornings
        """)

    with col2:
        st.markdown("""
            ### 🏥 স্বাস্থ্য সমস্যা (Health Issues)
            - **উচ্চ রক্তচাপ** → Very common (30%+ adults)
            - **ডায়াবেটিস** → Type 2 diabetes epidemic
            - **থাইরয়েড** → Increasingly common in women
            - **মোটা হওয়া** → Central obesity (পেটের মেদ)
            - **কিডনি সমস্যা** → Often linked with diabetes & BP

            ### 🌍 পরিবেশগত (Environmental)
            - **বায়ু দূষণ** → Dhaka/Chittagong air quality
            - **আর্সেনিক পানি** → Groundwater contamination
            - **তাপমাত্রা** → Extreme heat stress
        """)

    st.markdown("---")

    st.subheader("📊 বাংলাদেশে হৃদরোগের সাধারণ মান (Typical Values in Bangladesh)")

    st.markdown("""
        | পরামিতি | স্বাভাবিক | ঝুঁকিপূর্ণ | বাংলাদেশে সাধারণ |
        |---------|-----------|------------|-------------------|
        | **বয়স** | <40 | >50 | রোগীরা 45-55 বছরেই আসে |
        | **রক্তচাপ** | 120/80 | >140/90 | অনেকের 150-160/90-100 |
        | **কলেস্টেরল** | <200 | >240 | 220-280 সাধারণ |
        | **ব্লাড সুগার (ফাস্টিং)** | <100 | >126 | 110-180 প্রচলিত |
        | **ওজন (BMI)** | 18.5-24.9 | >25 | পুরুষ: 26-30, মহিলা: 25-28 |
        | **পেটের মেদ** | <90 cm (পুরুষ) | >100 cm | 95-110 cm প্রচলিত |
        | **ধূমপান** | না | হ্যাঁ | ৪০%+ পুরুষ ধূমপায়ী |
        | **শারীরিক পরিশ্রম** | 30 min/day | 0 min | অফিস-বাড়ি, গাড়িতেই যাতায়াত |
    """)

    st.markdown("---")

    st.subheader("🎯 ঝুঁকি যাচাইকরণ (Quick Risk Checklist)")

    risk_score = 0

    with st.expander("✅ নিজেকে যাচাই করুন (Check Yourself)"):
        q1 = st.checkbox("আপনার বয়স ৪৫ বছরের বেশি?")
        q2 = st.checkbox("আপনার পরিবারে হৃদরোগের ইতিহাস আছে?")
        q3 = st.checkbox("আপনি ধূমপান করেন বা আগে করতেন?")
        q4 = st.checkbox("আপনার রক্তচাপ ১৪০/৯০-এর বেশি?")
        q5 = st.checkbox("আপনার ডায়াবেটিস আছে?")
        q6 = st.checkbox("আপনার কলেস্টেরল ২০০-এর বেশি?")
        q7 = st.checkbox("আপনার পেটের মাপ ৩৬ ইঞ্চির বেশি?")
        q8 = st.checkbox("আপনি নিয়মিত হাঁটেন/ব্যায়াম করেন না?")
        q9 = st.checkbox("আপনি প্রতিদিন ১ চামচের বেশি নুন খান?")
        q10 = st.checkbox("আপনার মানসিক চাপ বেশি?")

        risk_score = sum([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])

        st.markdown("---")

        if risk_score == 0:
            st.success("🟢 **খুব কম ঝুঁকি** — চমৎকার! এভাবেই থাকুন।")
        elif risk_score <= 3:
            st.info(f"🟡 **মাঝারি ঝুঁকি ({risk_score}/10)** — সতর্ক থাকুন, নিয়মিত চেকআপ করুন।")
        elif risk_score <= 6:
            st.warning(f"🟠 **উচ্চ ঝুঁকি ({risk_score}/10)** — চিকিৎসকের পরামর্শ নিন, জীবনযাত্রা পরিবর্তন করুন।")
        else:
            st.error(f"🔴 **অত্যন্ত উচ্চ ঝুঁকি ({risk_score}/10)** — দ্রুত চিকিৎসকের সাথে কথা বলুন!")

    st.markdown("---")

    st.subheader("💡 বাংলাদেশের জন্য পরামর্শ (Recommendations for Bangladesh)")

    st.markdown("""
        <div class="warning-box">

        ### 🍽️ খাবারে পরিবর্তন
        - **ভাত কমান** → প্রতি খাবারে ১ কাপের বেশি নয়
        - **মাছ বাড়ান** → সপ্তাহে ৩-৪ দিন মাছ (ইলিশ, রুই, কাতল)
        - **তেল কমান** → রান্নায় সরিষার তেল ব্যবহার করুন, পরিমাণ কমান
        - **নুন কমান** → আচার, ভর্তা কম খান, টেবিলে নুন রাখবেন না
        - **সবজি বাড়ান** → শাক-সবজি প্রতি খাবারে

        ### 🏃‍♂️ শারীরিক পরিশ্রম
        - **হাঁটুন** → প্রতিদিন কমপক্ষে ৩০ মিনিট (ফজরের নামাজের পর হাঁটা ভালো)
        - **সিঁড়ি ব্যবহার করুন** → লিফট কম ব্যবহার করুন
        - **সাইকেল** → কাছাকাছি যাতায়াতে সাইকেল
        - **নামাজ** → ৫ ওয়াক্ত নামাজও শারীরিক পরিশ্রম

        ### 🧠 মানসিক স্বাস্থ্য
        - **ঘুম** → রাত ১১টার মধ্যে ঘুমাতে যান
        - **ধ্যান/জিকির** → দিনে ১০ মিনিট শান্ত বসুন
        - **পারিবারিক সময়** → পরিবারের সাথে সময় কাটান
        - **মোবাইল কম** → সোশ্যাল মিডিয়া কম ব্যবহার করুন

        ### 🏥 নিয়মিত চেকআপ
        - **বছরে ১বার** → পূর্ণাঙ্গ স্বাস্থ্য পরীক্ষা
        - **৩ মাসে ১বার** → রক্তচাপ ও ব্লাড সুগার
        - **৬ মাসে ১বার** → লিপিড প্রোফাইল (কলেস্টেরল)
        - **বিনামূল্যে** → সরকারি হাসপাতালে চেকআপ (নিয়মিত যান)

        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
        <div class="danger-box">
            <h4 style="color: inherit;">🚨 জরুরী চিকিৎসা সেবা (Emergency Services)</h4>
            <ul>
                <li><b>ন্যাশনাল হার্ট ফাউন্ডেশন:</b> 02-9122668</li>
                <li><b>স্কয়ার হাসপাতাল:</b> 02-8144466</li>
                <li><b>আনোয়ার খান মডার্ন মেডিকেল:</b> 02-9676356</li>
                <li><b>বঙ্গবন্ধু শেখ মুজিব মেডিকেল:</b> 02-55165001</li>
                <li><b>জরুরী: 999</b> (ন্যাশনাল হেল্পলাইন)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# About Page
elif page == "ℹ️ About":
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
        ### 🏥 Heart Disease Prediction System

        This application uses a **K-Nearest Neighbors (KNN)** machine learning model trained on the 
        [UCI Heart Disease dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) 
        to predict the likelihood of heart disease based on patient health parameters.

        ### 🔧 Model Details
        - **Algorithm**: K-Nearest Neighbors (KNN)
        - **Features**: 11 clinical parameters including age, blood pressure, cholesterol, etc.
        - **Preprocessing**: StandardScaler for feature normalization
        - **Encoding**: One-hot encoding for categorical variables

        ### 📊 Features Used
        | Feature | Description |
        |---------|-------------|
        | Age | Patient age in years |
        | Sex | Gender (M/F) |
        | ChestPainType | Type of chest pain (ATA, NAP, ASY, TA) |
        | RestingBP | Resting blood pressure (mm Hg) |
        | Cholesterol | Serum cholesterol (mg/dl) |
        | FastingBS | Fasting blood sugar > 120 mg/dl |
        | RestingECG | Resting electrocardiogram results |
        | MaxHR | Maximum heart rate achieved |
        | ExerciseAngina | Exercise-induced angina (Y/N) |
        | Oldpeak | ST depression induced by exercise |
        | ST_Slope | Slope of the peak exercise ST segment |

        ### ⚠️ Disclaimer
        <div class="danger-box">
            <b>Important:</b> This tool is for educational and screening purposes only. 
            It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider with any 
            questions you may have regarding a medical condition.
        </div>

        ### 👨‍💻 Technical Stack
        - **Frontend**: Streamlit
        - **Backend**: Python, Scikit-learn
        - **Visualization**: Matplotlib, Seaborn
        - **Model Format**: Joblib (PKL files)
    """, unsafe_allow_html=True)

    if model is not None:
        st.success("✅ System Status: All components loaded successfully")
    else:
        st.error("❌ System Status: Model files missing")