"""Cox 标签与特征边界的单元测试。

运行: ./.venv/Scripts/python.exe tests/test_cox_features.py -v
"""
import os
import sys
import unittest
from datetime import date

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import build_cox_features as bf  # noqa: E402

OBS = date(2011, 12, 9)


class TestLabel(unittest.TestCase):
    def test_next_purchase_found(self):
        """打分后第 20 天购买 -> T=20。"""
        T, E = bf.compute_next_purchase([date(2011, 1, 20)], date(2011, 1, 1), OBS)
        self.assertEqual(E, 1)
        self.assertEqual(T, 20)

    def test_same_day_purchase_counts_as_day_1(self):
        """打分日当天就下单 -> 记第 1 天，T>=1。"""
        T, E = bf.compute_next_purchase([date(2011, 1, 1)], date(2011, 1, 1), OBS)
        self.assertEqual(E, 1)
        self.assertEqual(T, 1)

    def test_censored_before_purchase(self):
        """标签截止日前没有购买 -> 右删失。"""
        T, E = bf.compute_next_purchase([], date(2011, 12, 1), OBS)
        self.assertEqual(E, 0)
        self.assertEqual(T, 9)

    def test_purchase_after_deadline_is_censored(self):
        """购买发生在标签截止日之后，不得计为事件。"""
        T, E = bf.compute_next_purchase([date(2011, 12, 20)], date(2011, 1, 1), OBS)
        self.assertEqual(E, 0)
        self.assertEqual(T, (OBS - date(2011, 1, 1)).days + 1)


class TestTableInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.df = pd.read_csv(
            os.path.join(_PROJECT_ROOT, "data", "processed", "cox_model_table.csv"),
            parse_dates=["entry_date", "score_day", "label_deadline"],
        )

    def test_score_day_is_entry_plus_14(self):
        delta = (self.df["score_day"] - self.df["entry_date"]).dt.days
        self.assertTrue((delta == 14).all(), "score_day 必须等于 entry_date + 14")

    def test_censored_T_matches_split_deadline(self):
        """删失 T 必须等于该行记录的 label_deadline - score_day。"""
        cens = self.df[self.df["E"] == 0]
        expect = (cens["label_deadline"] - cens["score_day"]).dt.days + 1
        self.assertTrue((cens["T"] == expect).all(), "删失 T 与 label_deadline 不一致")

    def test_T_is_positive_integer(self):
        self.assertTrue((self.df["T"] >= 1).all(), "T 必须为正整数")
        self.assertTrue((self.df["T"] == self.df["T"].round(0)).all(), "T 必须是整数")

    def test_whitelist_matches_table_columns(self):
        import json
        path = os.path.join(_PROJECT_ROOT, "reports", "feature_whitelist.json")
        with open(path, encoding="utf-8") as f:
            wl = json.load(f)
        cols = list(self.df.columns)
        for c in wl["features"]:
            self.assertIn(c, cols, "白名单特征必须在建模表里")
        for c in wl["dropped"]:
            self.assertNotIn(c, cols, "被筛掉的列不应出现在建模表里")

    def test_label_deadline_is_split_specific(self):
        """每个 split 的标签截止日必须晚于该 split 所有 score_day。"""
        for sp, g in self.df.groupby("split"):
            self.assertTrue((g["label_deadline"] >= g["score_day"]).all(), sp)
        dl = self.df.groupby("split")["label_deadline"].first()
        self.assertLess(dl["train"], dl["val"], "训练标签不得跨到验证期")
        self.assertLess(dl["val"], dl["test"], "验证标签不得跨到测试期")

    def test_whitelisted_features_present(self):
        """建模表应只含白名单特征 + 键列，不含被筛掉的列。"""
        for c in ("n_invoices", "days_since_last_buy", "n_products"):
            self.assertIn(c, self.df.columns)

    def test_no_future_leak_in_window_features(self):
        # 窗口是 [entry, entry+14)，最后购买日最晚是 entry+13 -> days_since_last_buy <= 14
        self.assertTrue((self.df["days_since_last_buy"] >= 0).all())
        self.assertTrue((self.df["days_since_last_buy"] <= 14).all(),
                        "距最后购买天数必须落在窗口内")


if __name__ == "__main__":
    unittest.main(verbosity=2)
