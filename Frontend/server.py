"""Heart Disease Prediction System: web server.

Serves the HTML/CSS/JS interface in ./static and exposes a small JSON API.
Every prediction is made here with the real model files, KNN_HeartPKL.pkl, scaler.pkl
and columns.pkl, exactly as the Streamlit app loads them. Nothing about the model is
copied into the browser.

Run:
    pip install -r requirements.txt
    python server.py            # then open http://127.0.0.1:8000
"""
import io
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = ['KNN_HeartPKL.pkl', 'scaler.pkl', 'columns.pkl']

# ---------------------------------------------------------------------------
# Input schema. Ranges and defaults are the ones the Streamlit form uses.
# ---------------------------------------------------------------------------
INPUT_COLUMNS = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
CATEGORIES = {
    'Sex': ['M', 'F'],
    'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
    'FastingBS': [0, 1],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Up', 'Flat', 'Down'],
}
NUMERIC = {
    'Age': dict(min=1, max=120, step=1),
    'RestingBP': dict(min=0, max=300, step=1),
    'Cholesterol': dict(min=0, max=600, step=1),
    'MaxHR': dict(min=60, max=220, step=1),
    'Oldpeak': dict(min=-3.0, max=10.0, step=0.1),
}
DEFAULTS = {'Age': 45, 'Sex': 'M', 'ChestPainType': 'ATA', 'RestingBP': 120, 'Cholesterol': 200,
            'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 150, 'ExerciseAngina': 'N',
            'Oldpeak': 0.0, 'ST_Slope': 'Up'}
# Features shown in the "where this patient sits" panel, with the label used there
COMPARE = [('Age', 'Age'), ('RestingBP', 'Blood pressure'), ('Cholesterol', 'Cholesterol'),
           ('MaxHR', 'Max heart rate'), ('Oldpeak', 'Oldpeak')]


# ---------------------------------------------------------------------------
# Load the model files (same search order as app.py)
# ---------------------------------------------------------------------------
def find_dir(files, extra=()):
    candidates = [HERE, '.', '/app', '/mount/src', os.path.join(HERE, '..')] + list(extra)
    for path in candidates:
        if all(os.path.exists(os.path.join(path, f)) for f in files):
            return path
    return None


art_dir = find_dir(ARTIFACTS)
if art_dir is None:
    raise SystemExit('Model files not found. Put %s next to server.py.' % ', '.join(ARTIFACTS))

model = joblib.load(os.path.join(art_dir, 'KNN_HeartPKL.pkl'))
scaler = joblib.load(os.path.join(art_dir, 'scaler.pkl'))
columns = joblib.load(os.path.join(art_dir, 'columns.pkl'))
CLASSES = list(model.classes_)
POS = CLASSES.index(1)
K = int(model.n_neighbors)


def encode(df):
    """One-hot encode against the training columns in columns.pkl, then cast to int.

    Dummy columns are read from columns.pkl ('Sex_M' means Sex == 'M'), so every row is
    encoded the same way no matter what else is in the batch. get_dummies(drop_first=True)
    on a single row, or on a batch missing a category, drops the wrong levels.
    """
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            out[col] = df[col]
            continue
        base, _, level = col.rpartition('_')
        out[col] = (df[base].astype(str) == level).astype(int) if base in df.columns else 0
    return out[columns].astype(float).astype(int)


def run_model(df):
    scaled = scaler.transform(encode(df))
    proba = model.predict_proba(scaled)
    pred = model.predict(scaled)
    return scaled, pred, proba


# ---------------------------------------------------------------------------
# Dataset context: held-out metrics, distributions, neighbor profiles.
# Computed from heart.csv at startup with the notebook's preparation steps.
# Everything degrades gracefully if heart.csv is not available.
# ---------------------------------------------------------------------------
class Context:
    pass


def build_context():
    data_dir = find_dir(['heart.csv'])
    if data_dir is None:
        return None
    raw = pd.read_csv(os.path.join(data_dir, 'heart.csv'))
    ctx = Context()
    ctx.total = len(raw)

    # Distributions for the comparison strips use the recorded values (0 means not measured)
    ctx.dist = {}
    for key, _ in COMPARE:
        v = raw[key].astype(float)
        v = v[v > 0] if key in ('RestingBP', 'Cholesterol') else v
        lo, hi = float(np.floor(v.min())), float(np.ceil(v.max()))
        hist, _ = np.histogram(v, bins=24, range=(lo, hi))
        ctx.dist[key] = dict(lo=lo, hi=hi, bins=hist.tolist(), values=np.sort(v.values))

    # Same preparation as Heart_Disease.ipynb: zeros replaced by the mean, one-hot, 80/20 split
    df = raw.copy()
    for col in ('Cholesterol', 'RestingBP'):
        df[col] = df[col].replace(0, df.loc[df[col] != 0, col].mean()).round(2)
    enc = pd.get_dummies(df, drop_first=True).astype(int)
    X, y = enc.drop(columns='HeartDisease'), enc['HeartDisease']
    if list(X.columns) != list(columns):
        return ctx
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pred = model.predict(scaler.transform(xte))
    ctx.metrics = dict(
        accuracy=float(accuracy_score(yte, pred)), f1=float(f1_score(yte, pred)),
        precision=float(precision_score(yte, pred)), recall=float(recall_score(yte, pred)),
        n_test=len(xte))

    # Neighbor profiles are only shown when the rebuilt training set is the one the model holds
    fit_x = getattr(model, '_fit_X', None)
    ctx.train_rows = None
    if fit_x is not None and len(xtr) == len(fit_x) and \
            np.allclose(StandardScaler().fit(xtr).transform(xtr), fit_x):
        ctx.train_rows = df.loc[xtr.index].reset_index(drop=True)
    return ctx


CTX = build_context()
N_TRAIN = int(getattr(model, 'n_samples_fit_', 0)) or None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_single(payload):
    errors, row = [], {}
    for col in INPUT_COLUMNS:
        val = payload.get(col)
        if col in NUMERIC:
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors.append('%s must be a number' % col)
                continue
            lim = NUMERIC[col]
            if not (lim['min'] <= num <= lim['max']):
                errors.append('%s must be from %s to %s' % (col, lim['min'], lim['max']))
            row[col] = num
        else:
            allowed = CATEGORIES[col]
            if col == 'FastingBS':
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    pass
            if val not in allowed:
                errors.append('%s must be one of %s' % (col, ', '.join(map(str, allowed))))
            row[col] = val
    return row, errors


class BatchError(Exception):
    pass


def read_batch(file_storage):
    if file_storage is None:
        raise BatchError('No file was uploaded.')
    data = file_storage.read()
    try:
        df = pd.read_csv(io.BytesIO(data), encoding='utf-8-sig', skipinitialspace=True)
    except pd.errors.EmptyDataError:
        raise BatchError('This file is empty.')
    except Exception as exc:
        raise BatchError('The file could not be read as CSV: %s' % exc)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        raise BatchError('This file has a header but no patient rows.')
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise BatchError('Missing columns: %s' % ', '.join(missing))
    for col in INPUT_COLUMNS:
        if col in CATEGORIES and col != 'FastingBS':
            df[col] = df[col].astype(str).str.strip()
            bad = ~df[col].isin(CATEGORIES[col])
        else:
            num = pd.to_numeric(df[col], errors='coerce')
            bad = num.isna()
            if col == 'FastingBS':
                bad = bad | ~num.isin(CATEGORIES[col])
            df[col] = num
        if bad.any():
            i = int(np.flatnonzero(bad.values)[0])
            shown = df[col].iloc[i]
            expect = ('Expected %s.' % ', '.join(map(str, CATEGORIES[col]))) if col in CATEGORIES \
                else 'That is not a number.'
            raise BatchError('Row %d: %s is "%s". %s' % (i + 1, col, '' if pd.isna(shown) else shown, expect))
    return df


def records(df):
    return [{k: (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v)) for k, v in r.items()}
            for r in df.to_dict(orient='records')]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=os.path.join(HERE, 'static'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024


@app.after_request
def no_cache(resp):
    if request.path == '/' or request.path.endswith(('.html', '.js', '.css')):
        resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='This file is larger than 20 MB.'), 413


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.get('/api/info')
def info():
    return jsonify(
        model=dict(name='K-nearest neighbors', k=K, n_train=N_TRAIN, classes=[int(c) for c in CLASSES]),
        metrics=getattr(CTX, 'metrics', None),
        dataset=dict(total=CTX.total) if CTX else None,
        input_columns=INPUT_COLUMNS, categories=CATEGORIES, numeric=NUMERIC, defaults=DEFAULTS,
        distributions={k: dict(lo=d['lo'], hi=d['hi'], bins=d['bins']) for k, d in CTX.dist.items()} if CTX else {},
        has_neighbor_profiles=bool(CTX and getattr(CTX, 'train_rows', None) is not None),
    )


@app.post('/api/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    row, errors = validate_single(payload)
    if errors:
        return jsonify(error='Check the highlighted values.', details=errors), 400

    df = pd.DataFrame([row])[INPUT_COLUMNS]
    scaled, pred, proba = run_model(df)
    risk = float(proba[0][POS])
    votes = int(round(risk * K))

    dist, idx = model.kneighbors(scaled, n_neighbors=K, return_distance=True)
    y_fit = getattr(model, '_y', None)
    rows = CTX.train_rows if CTX is not None and getattr(CTX, 'train_rows', None) is not None else None
    neighbors = []
    for d, i in zip(dist[0], idx[0]):
        item = dict(distance=float(d), label=int(CLASSES[int(y_fit[i])]) if y_fit is not None else None)
        if rows is not None:
            r = rows.iloc[int(i)]
            item['profile'] = {c: (r[c].item() if hasattr(r[c], 'item') else r[c])
                               for c in ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                                         'MaxHR', 'Oldpeak', 'ST_Slope')}
        neighbors.append(item)

    compare = []
    if CTX is not None:
        for key, label in COMPARE:
            vals = CTX.dist[key]['values']
            compare.append(dict(key=key, label=label, value=row[key],
                                percentile=float((vals <= row[key]).mean() * 100)))

    return jsonify(prediction=int(pred[0]), risk=risk, confidence=float(proba[0].max()),
                   votes=votes, k=K, neighbors=neighbors, compare=compare)


@app.post('/api/batch/validate')
def batch_validate():
    try:
        df = read_batch(request.files.get('file'))
    except BatchError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(rows=len(df), columns=list(df.columns), preview=records(df[INPUT_COLUMNS].head(5)))


@app.post('/api/batch/predict')
def batch_predict():
    try:
        df = read_batch(request.files.get('file'))
    except BatchError as exc:
        return jsonify(error=str(exc)), 400

    _, pred, proba = run_model(df[INPUT_COLUMNS])
    out = df.copy()
    out['Prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in pred]
    out['Confidence'] = proba.max(axis=1) * 100
    out['Risk_Score'] = proba[:, POS] * 100

    rows = records(out)
    votes = np.rint(proba[:, POS] * K).astype(int)
    for r, v in zip(rows, votes):
        r['Votes'] = int(v)
    at_risk = int((pred == 1).sum())
    return jsonify(
        k=K, rows=rows, input_columns=INPUT_COLUMNS,
        summary=dict(total=len(out), at_risk=at_risk, healthy=len(out) - at_risk),
        votes_histogram=np.bincount(votes, minlength=K + 1).tolist(),
        csv=out.to_csv(index=False))


@app.get('/api/sample')
def sample():
    path = os.path.join(HERE, 'bangladesh_heart_patients.csv')
    if not os.path.exists(path):
        return jsonify(error='The sample file is not on the server.'), 404
    return send_from_directory(HERE, 'bangladesh_heart_patients.csv', mimetype='text/csv')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print('Serving on http://127.0.0.1:%d' % port)
    app.run(host='127.0.0.1', port=port, debug=False)