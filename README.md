# ❤️ Heart Disease Prediction System

> 🏥 Heart disease risk assessment using Machine Learning (K-Nearest Neighbors)
---

## 🚀 Live Demo

**Try the app here:** 👉 **[https://predictheart-disease.streamlit.app/](https://predictheart-disease.streamlit.app/)**
---
## 📖 Overview

This project is a **Heart Disease Prediction System** that uses machine learning to assess the risk of heart disease based on patient health parameters. The system is built with a **Streamlit frontend** for easy interaction and a **KNN (K-Nearest Neighbors)** model trained on the UCI Heart Disease dataset.

The application provides:
- 🔮 **Single Patient Prediction** — Input individual health data and get instant risk assessment
- 📊 **Batch Prediction** — Upload a CSV file to predict multiple patients at once
- 📈 **Visual Analytics** — Feature analysis charts and risk distribution plots
- 💾 **Downloadable Results** — Export prediction results as CSV

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Single Prediction** | Interactive form to input patient vitals and clinical features |
| **Batch Prediction** | Upload CSV files with multiple patient records |
| **Confidence Score** | Model confidence percentage with progress bar |
| **Feature Analysis** | Horizontal bar chart showing input feature values |
| **Risk Distribution** | Pie chart and histogram for batch predictions |
| **Dark Mode Support** | Fully compatible with Streamlit's light/dark themes |
| **Responsive UI** | Clean, mobile-friendly interface |
| **Auto Model Loading** | Smart path detection for local and cloud deployment |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Backend** | Python 3.8+ |
| **ML Library** | [scikit-learn](https://scikit-learn.org/) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Model Serialization** | [Joblib](https://joblib.readthedocs.io/) |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/soumensen411/Heart_Disease.git
cd Heart_Disease
```

### Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
joblib>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 🎯 Usage

### Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use

1. **Navigate** to the desired page from the sidebar:
   - 🏠 **Home** — Overview and instructions
   - 🔮 **Predict** — Single patient prediction
   - 📊 **Batch Prediction** — Multiple patients via CSV upload

2. **Input Data** — Fill in the health parameters or upload a CSV file

3. **Get Results** — Click "Predict" to see the risk assessment with confidence score

### Input Features

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| Age | Patient age in years | 1-120 |
| Sex | Gender | M / F |
| ChestPainType | Type of chest pain | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure | 0-300 mm Hg |
| Cholesterol | Serum cholesterol | 0-600 mg/dl |
| FastingBS | Fasting blood sugar > 120 | 0 (No) / 1 (Yes) |
| RestingECG | Resting ECG results | Normal, ST, LVH |
| MaxHR | Maximum heart rate | 60-220 |
| ExerciseAngina | Exercise-induced angina | N / Y |
| Oldpeak | ST depression | -3.0 to 10.0 |
| ST_Slope | ST segment slope | Up, Flat, Down |

---

## 🤖 Model Details

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Training Dataset:** [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features:** 11 clinical parameters
- **Preprocessing:**
  - One-hot encoding for categorical variables
  - StandardScaler for feature normalization
- **Model Files:**
  - `KNN_HeartPKL.pkl` — Trained KNN model
  - `scaler.pkl` — StandardScaler fitted on training data
  - `columns.pkl` — Feature column names for encoding alignment

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~85-90% |
| F1 Score | ~0.84 |

> ⚠️ **Note:** This is a screening tool for educational purposes, not a substitute for professional medical diagnosis.
---

## 📸 Screenshots

### 🏠 Home Page
![Home Page](https://ik.imagekit.io/xrrsiofbq/Data%20Science/homepage)

### 🔮 Single Prediction
![Single Prediction](https://ik.imagekit.io/xrrsiofbq/Data%20Science/singlePrediction)

### 📊 Batch Prediction Results
![Batch Prediction](https://ik.imagekit.io/xrrsiofbq/Data%20Science/BatchPredictionResults)

---
## 👨‍💻 Author

**Soumen Sen**

- **GitHub:** [@soumensen411](https://github.com/soumensen411)
- **Project:** [Heart Disease Prediction](https://github.com/soumensen411/Heart_Disease)
- **Live App:** [predictheart-disease.streamlit.app](https://predictheart-disease.streamlit.app/)
---
<p align="center">
  Made with ❤️ for better healthcare
</p>

