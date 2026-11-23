import sys
import unittest
from pathlib import Path

import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_cox as ev


class TestHeldOutEvaluation(unittest.TestCase):
    def test_window_includes_day_30_and_excludes_all_short_followup(self):
        day = pd.Timestamp("2011-01-01")
        frame = pd.DataFrame({
            "score_day": day,
            "label_deadline": day + pd.to_timedelta([29, 29, 28, 28, 40], unit="D"),
            "T": [30, 30, 29, 1, 31],
            "E": [1, 0, 0, 1, 1],
        })
        result = ev.horizon_cohort(frame, 30)
        self.assertEqual(result["purchased"].to_dict(), {0: True, 1: False, 4: False})

    def test_observed_bin_survival_matches_km_with_censoring_at_boundary(self):
        cohort = pd.DataFrame({"T": [5, 30, 30, 45], "E": [1, 1, 0, 1],
                               "purchased": [True, True, False, False]}, index=[3, 8, 9, 11])
        predicts = pd.Series(0.75, index=[11, 9, 8, 3])
        bins = ev.calibration(cohort, predicts, ev.COX, 30)
        self.assertEqual(len(bins), 1)
        row = bins.iloc[0]
        km = KaplanMeierFitter().fit(cohort["T"], cohort["E"])
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["predicted_survival"], 0.75)
        self.assertAlmostEqual(row["observed_survival"], km.predict(30))
        self.assertLess(row["ci_low"], 0.5)
        self.assertGreater(row["ci_high"], 0.5)

    def test_ranks_earlier_purchases_first_and_ignores_timing_after_30(self):
        cohort = pd.DataFrame({"T": [30, 4, 40, 90], "purchased": [True, True, False, False]})
        probabilities = pd.Series({3: 0.1, 2: 0.2, 1: 0.8, 0: 0.5})
        best = ev.fixed_scores(cohort, probabilities)
        self.assertEqual(best["C@30"], 1.0)
        self.assertEqual(best["ROC AUC"], 1.0)
        swapped = ev.fixed_scores(cohort, probabilities.replace({0.8: 0.5, 0.5: 0.8}))
        self.assertLess(swapped["C@30"], best["C@30"])
        self.assertEqual(swapped["ROC AUC"], best["ROC AUC"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
