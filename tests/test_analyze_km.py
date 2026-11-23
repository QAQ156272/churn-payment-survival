"""analyze_km 的分组提取与检验决策测试。

运行: ./.venv/Scripts/python.exe tests/test_analyze_km.py -v
"""
import os
import sys
import unittest
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import analyze_km as km  # noqa: E402


class TestExtractFirstInvoice(unittest.TestCase):
    def test_same_invoice_multiple_lines_sums(self):
        """同一张发票的多行明细要累加。"""
        rows = [
            ("INV1", 2, datetime(2011, 1, 1, 10), 5.0, 1),
            ("INV1", 3, datetime(2011, 1, 1, 10), 2.0, 1),
        ]
        df = km.extract_first_invoice(rows, {1})
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["first_invoice_amount"], 16.0)

    def test_same_day_multiple_invoices_takes_earliest_only(self):
        """同日多张发票只取最早一张，后续发票不得累加。"""
        rows = [
            ("INV1", 1, datetime(2011, 1, 1, 9), 100.0, 1),
            ("INV2", 1, datetime(2011, 1, 1, 15), 900.0, 1),
        ]
        df = km.extract_first_invoice(rows, {1})
        self.assertEqual(df.iloc[0]["invoice"], "INV1")
        self.assertAlmostEqual(df.iloc[0]["first_invoice_amount"], 100.0)
        self.assertEqual(int(df.iloc[0]["tied_time_invoices"]), 1)

    def test_tied_time_stable_by_invoice_number(self):
        """时间并列时按发票号稳定排序，并报告并列张数。"""
        rows = [
            ("INV9", 1, datetime(2011, 1, 1, 9), 50.0, 1),
            ("INV2", 1, datetime(2011, 1, 1, 9), 70.0, 1),
        ]
        df = km.extract_first_invoice(rows, {1})
        self.assertEqual(df.iloc[0]["invoice"], "INV2")
        self.assertEqual(int(df.iloc[0]["tied_time_invoices"]), 2)
        self.assertAlmostEqual(df.iloc[0]["first_invoice_amount"], 70.0)

    def test_invalid_lines_ignored(self):
        """Quantity<=0 / Price<=0 的明细不参与聚合。"""
        rows = [
            ("INV1", 0, datetime(2011, 1, 1, 9), 10.0, 1),
            ("INV1", 2, datetime(2011, 1, 1, 9), 0.0, 1),
            ("INV2", 2, datetime(2011, 1, 1, 9), 3.0, 1),
        ]
        df = km.extract_first_invoice(rows, {1})
        self.assertEqual(df.iloc[0]["invoice"], "INV2")
        self.assertAlmostEqual(df.iloc[0]["first_invoice_amount"], 6.0)

    def test_user_filter(self):
        rows = [
            ("INV1", 1, datetime(2011, 1, 1, 9), 10.0, 1),
            ("INV2", 1, datetime(2011, 1, 1, 9), 20.0, 2),
        ]
        df = km.extract_first_invoice(rows, {1})
        self.assertEqual(len(df), 1)


class TestGrouping(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(km.assign_group(200.0), 0)
        self.assertEqual(km.assign_group(200.01), 1)
        self.assertEqual(km.assign_group(500.0), 1)
        self.assertEqual(km.assign_group(500.01), 2)

    def test_decide_test_mode(self):
        self.assertEqual(km.decide_test_mode([]), "not_tested")
        self.assertEqual(km.decide_test_mode([1]), "not_tested")
        self.assertEqual(km.decide_test_mode([0, 1]), "two_group")
        self.assertEqual(km.decide_test_mode([0, 1, 2]), "multi_group")


class TestRawMissing(unittest.TestCase):
    def test_missing_raw_raises(self):
        old = km.RAW
        km.RAW = os.path.join(_PROJECT_ROOT, ".tmp", "does_not_exist.xlsx")
        try:
            with self.assertRaises(FileNotFoundError):
                km.load_first_invoice({1})
        finally:
            km.RAW = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
