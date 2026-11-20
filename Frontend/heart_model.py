"""Single source of truth for the model's preprocessing.

Everything that happens between a raw patient record and the KNN lives in one sklearn
Pipeline, so training, the Flask server and the Streamlit app cannot drift apart:

  raw row -> clean (0 = missing -> NaN) -> impute + scale numbers, one-hot categories -> KNN

The pipeline is saved as ONE file (heart_model.pkl). There is no separate scaler.pkl or
columns.pkl to get out of sync, and one-hot encoding uses fixed category lists, so a single
row is encoded exactly like a row in a big batch.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

INPUT_COLUMNS = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
NUMERIC = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CATEGORICAL = {
    'Sex': ['F', 'M'],
    'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
    'FastingBS': [0, 1],
    'RestingECG': ['LVH', 'Normal', 'ST'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Down', 'Flat', 'Up'],
}
ZERO_MEANS_MISSING = ['RestingBP', 'Cholesterol']  # 0 mm Hg / 0 mg/dl is not a real reading
K = 5


def clean(df):
    """Keep only the model inputs, mark impossible zeros as missing, fix dtypes."""
    df = df[INPUT_COLUMNS].copy()
    for col in ZERO_MEANS_MISSING:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(0, np.nan)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    df['FastingBS'] = pd.to_numeric(df['FastingBS']).astype(int)
    return df


def build_pipeline(k=K):
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUMERIC),
        ('cat', OneHotEncoder(categories=list(CATEGORICAL.values()), handle_unknown='ignore'),
         list(CATEGORICAL)),
    ])
    return Pipeline([('clean', FunctionTransformer(clean)),
                     ('prep', prep),
                     ('model', KNeighborsClassifier(n_neighbors=k))])