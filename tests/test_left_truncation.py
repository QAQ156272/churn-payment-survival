import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import check_left_truncation as lt  # noqa: E402


class TestContamination(unittest.TestCase):
    def setUp(self):
        self.days = pd.DataFrame({
            "user_id": [1, 1, 2, 3, 3],
            "date": pd.to_datetime(["2010-01-01", "2010-12-05", "2010-12-10",
                                    "2011-03-01", "2011-04-01"]),
        })

    def test_returning_flag_uses_only_pre_start_history(self):
        table = lt.contamination(self.days, pd.Timestamp("2010-12-01"))
        self.assertEqual(table.loc[1, "returning"], True)   # 伪起点前买过
        self.assertEqual(table.loc[2, "returning"], False)
        self.assertEqual(table.loc[3, "days_since_start"], 90)  # 取伪起点后最早一天

    def test_lookback_keeps_only_late_first_seen(self):
        table = lt.contamination(self.days, pd.Timestamp("2010-12-01"))
        out = lt.by_lookback(table, lookbacks=(30,)).iloc[0]
        self.assertEqual(out["apparent_new"], 1)
        self.assertEqual(out["returning_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
