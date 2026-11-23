import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import check_cox_ph as ph


class TestPHExperiment(unittest.TestCase):
    def test_holm_monotone_in_rank_and_restores_original_order(self):
        np.testing.assert_allclose(ph.holm([0.80, 0.01, 0.04, 0.035]), [0.80, 0.04, 0.105, 0.105])

    def test_global_time_rank_is_independent_of_row_order_and_averages_ties(self):
        t = pd.Series([8, 12, 2, 8, 1], index=[11, 12, 21, 22, 23])
        events = pd.Series([True, False, True, True, False], index=t.index)
        ranks = ph.event_midranks(t, events, None)
        self.assertEqual(ranks[events].to_dict(), {11: 2.5, 21: 1.0, 22: 2.5})
        reversed_ranks = ph.event_midranks(t.iloc[::-1], events.iloc[::-1], None)
        pd.testing.assert_series_equal(ranks, reversed_ranks.reindex(t.index))

    def test_scoring_with_equal_covariates_uses_correct_stratum_baseline(self):
        rng = np.random.default_rng(123)
        frame = pd.DataFrame({
            "T": np.r_[rng.integers(5, 160, 100), rng.integers(45, 320, 100)],
            "E": rng.binomial(1, 0.8, 200).astype(bool),
            "x": rng.normal(size=200),
            ph.tc.BASELINE: np.repeat([13, 14], 100),
        })
        frame = ph.with_stratum(frame).drop(columns=ph.tc.BASELINE)
        model = CoxPHFitter(strata=[ph.STRATUM]).fit(frame, "T", "E")
        predictors = pd.DataFrame({"x": [0.0, 0.0, 0.0],
                                   ph.STRATUM: ["14", "1-13", "14"]}, index=[21, 4, 9])
        scores = ph.survival_score(model, predictors)
        self.assertEqual(scores.index.tolist(), [21, 4, 9])
        self.assertGreater(scores[21], scores[4])
        self.assertEqual(scores[21], scores[9])
        diagnostics = ph.diagnose(model, frame, "stratified")
        self.assertEqual(diagnostics["feature"].tolist(), ["x"] * len(ph.TRANSFORMS))
        self.assertTrue(diagnostics["p_holm"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
