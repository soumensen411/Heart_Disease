"""Run with:  python test_model.py   (or: python -m pytest test_model.py)"""
import io
import os
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
import heart_model  # noqa: E402
from heart_model import INPUT_COLUMNS  # noqa: E402

RAW = pd.read_csv(os.path.join(HERE, 'heart.csv'))
PIPE = joblib.load(os.path.join(HERE, 'heart_model.pkl'))
BASE = dict(Age=45, Sex='M', ChestPainType='ATA', RestingBP=120, Cholesterol=200, FastingBS=0,
            RestingECG='Normal', MaxHR=150, ExerciseAngina='N', Oldpeak=0.0, ST_Slope='Up')
HIGH = dict(Age=65, Sex='M', ChestPainType='ASY', RestingBP=160, Cholesterol=280, FastingBS=1,
            RestingECG='ST', MaxHR=100, ExerciseAngina='Y', Oldpeak=3.0, ST_Slope='Flat')


class PipelineTests(unittest.TestCase):
    def test_single_row_equals_batch(self):
        """The old Streamlit bug: one row was encoded differently from the same row in a batch."""
        X = RAW[INPUT_COLUMNS].head(200)
        batch = PIPE.predict_proba(X)
        single = np.vstack([PIPE.predict_proba(X.iloc[[i]]) for i in range(len(X))])
        np.testing.assert_allclose(single, batch)

    def test_every_category_is_encoded(self):
        """A single row must light up exactly one dummy per category, never all zeros."""
        enc = PIPE[:-1].transform(pd.DataFrame([BASE]))
        names = PIPE.named_steps['prep'].get_feature_names_out()
        hot = {n for n, v in zip(names, np.ravel(enc)) if n.startswith('cat__') and v == 1}
        self.assertEqual(hot, {'cat__Sex_M', 'cat__ChestPainType_ATA', 'cat__FastingBS_0',
                               'cat__RestingECG_Normal', 'cat__ExerciseAngina_N', 'cat__ST_Slope_Up'})

    def test_clinical_sanity(self):
        self.assertEqual(PIPE.predict(pd.DataFrame([BASE]))[0], 0)
        self.assertEqual(PIPE.predict(pd.DataFrame([HIGH]))[0], 1)

    def test_zero_cholesterol_treated_as_missing(self):
        a = PIPE.predict_proba(pd.DataFrame([dict(BASE, Cholesterol=0, RestingBP=0)]))
        median = (RAW.loc[RAW.Cholesterol > 0, 'Cholesterol'].median(),
                  RAW.loc[RAW.RestingBP > 0, 'RestingBP'].median())
        b = PIPE.predict_proba(pd.DataFrame([dict(BASE, Cholesterol=median[0], RestingBP=median[1])]))
        np.testing.assert_allclose(a, b)

    def test_decimals_not_truncated(self):
        a = PIPE[:-1].transform(pd.DataFrame([dict(BASE, Oldpeak=0.9)]))
        b = PIPE[:-1].transform(pd.DataFrame([dict(BASE, Oldpeak=0.0)]))
        self.assertFalse(np.allclose(a, b))

    def test_column_order_and_extra_columns_ignored(self):
        shuffled = pd.DataFrame([BASE])[list(reversed(INPUT_COLUMNS))].assign(HeartDisease=1, Junk='x')
        np.testing.assert_allclose(PIPE.predict_proba(shuffled), PIPE.predict_proba(pd.DataFrame([BASE])))

    def test_accuracy_floor_cross_validated(self):
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        s = cross_val_score(heart_model.build_pipeline(), RAW[INPUT_COLUMNS], RAW.HeartDisease,
                            cv=StratifiedKFold(5, shuffle=True, random_state=0))
        self.assertGreater(s.mean(), 0.84, s)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.server, cls.c = server, server.app.test_client()

    def post(self, payload):
        return self.c.post('/api/predict', json=payload)

    def test_predict_endpoints(self):
        lo, hi = self.post(BASE).get_json(), self.post(HIGH).get_json()
        self.assertEqual((lo['prediction'], hi['prediction']), (0, 1))
        self.assertEqual(len(lo['neighbors']), 5)
        self.assertIn('profile', lo['neighbors'][0])

    def test_api_matches_pipeline(self):
        r = self.post(HIGH).get_json()
        self.assertAlmostEqual(r['risk'], PIPE.predict_proba(pd.DataFrame([HIGH]))[0][1])

    def test_batch_matches_single(self):
        csv = pd.DataFrame([BASE, HIGH]).to_csv(index=False).encode()
        r = self.c.post('/api/batch/predict', data={'file': (io.BytesIO(csv), 't.csv')},
                        content_type='multipart/form-data').get_json()
        self.assertEqual([x['Prediction'] for x in r['rows']], ['No Heart Disease', 'Heart Disease'])

    def test_validation(self):
        self.assertEqual(self.post(dict(BASE, Age=500)).status_code, 400)
        self.assertEqual(self.post(dict(BASE, Sex='X')).status_code, 400)

    def test_info_metrics(self):
        m = self.c.get('/api/info').get_json()['metrics']
        self.assertGreater(m['accuracy'], 0.84)


if __name__ == '__main__':
    unittest.main(verbosity=2)