"""Train the heart disease model and save it.

    python train.py            # reads ../heart.csv (or ./heart.csv), writes heart_model.pkl + metrics.json

Honest evaluation: metrics come from 5-fold cross-validation repeated 3 times, with the
whole pipeline (imputer, scaler, encoder, KNN) re-fitted inside every fold, so nothing from
a validation fold leaks into training. The saved model is then fitted on all the data.
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

from heart_model import INPUT_COLUMNS, build_pipeline

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))


def find_csv():
    for p in (os.path.join(HERE, 'heart.csv'), os.path.join(HERE, '..', 'heart.csv')):
        if os.path.exists(p):
            return p
    raise SystemExit('heart.csv not found next to train.py or in the parent folder.')


def main():
    raw = pd.read_csv(find_csv())
    X, y = raw[INPUT_COLUMNS], raw['HeartDisease']

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = cross_val_score(build_pipeline(), X, y, cv=cv, scoring='accuracy')
    print('Repeated 5-fold CV accuracy: %.4f +/- %.4f' % (scores.mean(), scores.std()))

    # One out-of-fold prediction per patient, for precision / recall / F1
    oof = cross_val_predict(build_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    metrics = dict(
        accuracy=float(accuracy_score(y, oof)), f1=float(f1_score(y, oof)),
        precision=float(precision_score(y, oof)), recall=float(recall_score(y, oof)),
        n_test=int(len(y)), cv_accuracy_mean=float(scores.mean()), cv_accuracy_std=float(scores.std()),
        method='5-fold cross-validated predictions')
    print('Out-of-fold:', {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    model = build_pipeline().fit(X, y)
    joblib.dump(model, os.path.join(HERE, 'heart_model.pkl'))
    with open(os.path.join(HERE, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved heart_model.pkl and metrics.json (trained on all %d records)' % len(y))


if __name__ == '__main__':
    main()