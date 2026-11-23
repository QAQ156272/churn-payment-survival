import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import build_warning, evaluate_cox, train_cox, warning


class TestWarningHandoff(unittest.TestCase):
    def test_portable_curves_match_fitted_cox(self):
        rng = np.random.default_rng(241)
        train = warning.demo_snapshot()
        train["T"] = rng.integers(1, 100, len(train))
        train["E"] = rng.binomial(1, 0.7, len(train))
        feats = list(warning.FEATURE_LABELS)
        spec = json.loads(json.dumps(build_warning.build_model(train, feats)))
        snapshot = train.iloc[::13].copy()
        model, _ = train_cox.fit(train, feats)
        scored, curves = warning.score(snapshot, spec)
        expected = evaluate_cox.survival_at_days(model, snapshot[feats])
        np.testing.assert_allclose(curves[expected.columns], expected, rtol=1e-12)
        np.testing.assert_array_equal(scored["user_id"], snapshot["user_id"])

    def test_warning_list_caps_high_only_and_breaks_score_ties_by_id(self):
        ids = [f"u{i:04d}" for i in range(1002)]
        scored = pd.DataFrame({"user_id": ids[::-1], "risk_score": 0.9, "risk_tier": "高"})
        scored.loc[scored["user_id"].eq("u0000"), ["risk_tier", "risk_score"]] = ["中高", 0.95]
        result = warning.warning_list(scored)
        self.assertEqual(result["user_id"].tolist(), ids[1:1001])
        self.assertEqual(result["rank"].tolist(), list(range(1, 1001)))

    def test_snapshot_rejects_formula_ids_and_preserves_leading_zeros(self):
        frame = warning.demo_snapshot().iloc[:1].copy()
        frame["user_id"] = "0001"
        self.assertEqual(warning.validate_snapshot(frame, list(warning.FEATURE_LABELS)).at[0, "user_id"], "0001")
        frame["user_id"] = "=1+1"
        with self.assertRaises(ValueError):
            warning.validate_snapshot(frame, list(warning.FEATURE_LABELS))

    def test_shipped_demo_has_candidates_and_score_curve_agree(self):
        path = Path(__file__).resolve().parents[1] / "models" / "cox_warning.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        scored, curves = warning.score(warning.demo_snapshot(), spec)
        self.assertFalse(warning.warning_list(scored).empty)
        self.assertTrue(curves.diff(axis=1).iloc[:, 1:].le(0).all().all())
        np.testing.assert_array_equal(curves[30], scored["risk_score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
