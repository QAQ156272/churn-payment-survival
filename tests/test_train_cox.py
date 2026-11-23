import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_cox as tc


class TestTrainingInputs(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "user_id": [1, 2, 3],
            "entry_date": pd.to_datetime(["2010-12-18", "2011-01-18", "2011-02-15"]),
            "score_day": pd.to_datetime(["2011-01-01", "2011-02-01", "2011-03-01"]),
            "label_deadline": pd.to_datetime(["2011-01-31", "2011-02-28", "2011-03-31"]),
            "split": ["train", "val", "test"],
            "T": [31, 2, 3],
            "E": [0, 1, 1],
            tc.BASELINE: [14, 11, 6],
        })

    def test_valid_input_and_forbidden_features(self):
        self.assertEqual(tc.validate(self.data, [tc.BASELINE]), [])
        for column in ("T", "E", "user_id", "score_day"):
            with self.subTest(column=column):
                self.assertTrue(tc.validate(self.data, [tc.BASELINE, column]))

    def test_missing_and_overlapping_deadline(self):
        for value in (pd.NaT, pd.Timestamp("2011-02-01")):
            with self.subTest(value=value):
                data = self.data.copy()
                data.loc[0, "label_deadline"] = value
                self.assertTrue(tc.validate(data, [tc.BASELINE]))


class TestCoxFallback(unittest.TestCase):
    def test_singular_fit_recovers_with_l2(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=160)
        event_time = rng.exponential(scale=np.exp(-0.5 * x))
        censor_time = rng.exponential(scale=2, size=len(x))
        data = pd.DataFrame({
            "T": np.minimum(event_time, censor_time),
            "E": (event_time <= censor_time).astype(int),
            tc.BASELINE: x,
            "duplicate": x,
        })
        features = ([tc.BASELINE], [tc.BASELINE, "duplicate"])
        train, val = data.iloc[:100], data.iloc[100:]
        with self.assertRaises(ConvergenceError):
            tc.fit(train, features[1])
        selected, _ = tc.choose_fit(train, val, features)
        self.assertIsNotNone(selected)
        penalty, (_, results, scores) = selected
        self.assertGreater(penalty, 0)
        self.assertEqual(len(results), 3)
        self.assertTrue(np.isfinite(results[["HR", "CI_low", "CI_high"]]).all().all())
        self.assertTrue(all(0 <= score <= 1 for score in scores))


if __name__ == "__main__":
    unittest.main(verbosity=2)
