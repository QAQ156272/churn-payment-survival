"""QC 门禁的回归测试。

运行: ./.venv/Scripts/python.exe tests/test_build_survival.py -v

注意: 本机 DSH 沙箱下 tempfile.TemporaryDirectory() 创建的目录不可写，
所以测试自己用 os.makedirs 建工作目录，不依赖 tempfile。
"""
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TMP_ROOT = os.path.join(_PROJECT_ROOT, ".tmp", "test-tmp")
os.makedirs(_TMP_ROOT, exist_ok=True)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
import build_survival as bs  # noqa: E402

HEADER = "user_id,first_date,last_date\n"


def _df(first, last, uid=1):
    return pd.DataFrame({
        "user_id": [uid],
        "first_date": pd.to_datetime([first]),
        "last_date": pd.to_datetime([last]),
    })


class TestEventDefinition(unittest.TestCase):
    def test_gap_29_is_censored(self):
        """末购距观察终点 29 天 -> 还没到 30 天 -> 删失。"""
        out = bs.build(_df("2011-11-10", "2011-11-10"), 30)
        self.assertEqual(int(out["E"].iloc[0]), 0)

    def test_gap_30_is_event(self):
        """末购距观察终点正好 30 天 -> 事件。"""
        out = bs.build(_df("2011-11-09", "2011-11-09"), 30)
        self.assertEqual(int(out["E"].iloc[0]), 1)

    def test_same_day_purchase_gets_T1(self):
        """首末同日的事件，T 必须是 1 而不是 0。"""
        out = bs.build(_df("2011-01-01", "2011-01-01"), 30)
        self.assertEqual(int(out["E"].iloc[0]), 1)
        self.assertEqual(int(out["T"].iloc[0]), 1)

    def test_censored_T_counts_to_obs_end(self):
        """删失用户的 T 算到观察终点，而不是末购日。"""
        out = bs.build(_df("2011-12-01", "2011-12-05"), 30)
        self.assertEqual(int(out["E"].iloc[0]), 0)
        self.assertEqual(int(out["T"].iloc[0]), (bs.OBS_END - pd.Timestamp("2011-12-01")).days + 1)


class TestLookback(unittest.TestCase):
    def test_boundary_and_exclusion(self):
        """首次出现早于 COHORT_START 剔除；正好等于 COHORT_START 保留。"""
        start = bs.COHORT_START
        df = pd.concat([
            _df((start - pd.Timedelta(days=1)).date().isoformat(), "2011-06-01", uid=1),
            _df(start.date().isoformat(), "2011-06-01", uid=2),
        ], ignore_index=True)
        kept, n_excluded = bs.apply_lookback(df)
        self.assertEqual(n_excluded, 1)
        self.assertEqual(kept["user_id"].tolist(), [2])

    def test_lookback_matches_constant(self):
        self.assertEqual((bs.COHORT_START - bs.OBS_START).days, bs.LOOKBACK_DAYS)


class TestQCGate(unittest.TestCase):
    def _case(self, name):
        d = os.path.join(_TMP_ROOT, name)
        os.makedirs(d, exist_ok=True)
        return d

    def _run(self, inp_csv, out_path):
        old_in, old_out = bs.IN, bs.OUT
        bs.IN, bs.OUT = inp_csv, out_path
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                rc = bs.main()
        finally:
            bs.IN, bs.OUT = old_in, old_out
        return rc

    @staticmethod
    def _write(path, text):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def _read(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _assert_rejected(self, name, content):
        td = self._case(name)
        inp, out = os.path.join(td, "in.csv"), os.path.join(td, "out.csv")
        self._write(inp, content)
        self._write(out, "SENTINEL\n")

        rc = self._run(inp, out)

        self.assertNotEqual(rc, 0, f"{name}: 非法输入必须返回非零")
        self.assertEqual(self._read(out), "SENTINEL\n", f"{name}: 不得覆盖旧输出")

    # ---- 结构类非法输入 ----
    def test_duplicated_user_id_aborts(self):
        self._assert_rejected(
            "dup_user_id",
            HEADER + "1,2011-01-01,2011-01-05\n1,2011-01-02,2011-01-06\n",
        )

    def test_empty_header_only_aborts(self):
        self._assert_rejected("empty_input", HEADER)

    def test_missing_column_aborts(self):
        self._assert_rejected("missing_col", "user_id,first_date\n1,2011-01-01\n")

    def test_zero_byte_file_aborts(self):
        """零字节文件（连表头都没有）必须有明确诊断，而不是抛 EmptyDataError。"""
        self._assert_rejected("zero_bytes", "")

    # ---- 字段格式类非法输入 ----
    def test_space_only_user_id_aborts(self):
        self._assert_rejected("space_id", HEADER + "  ,2011-01-01,2011-01-05\n")

    def test_float_user_id_aborts(self):
        self._assert_rejected("float_id", HEADER + "1.5,2011-01-01,2011-01-05\n")

    def test_numeric_date_aborts(self):
        """20110101 会被 pandas 当成 Unix 时间戳，必须在解析前拦住。"""
        self._assert_rejected("numeric_date", HEADER + "1,20110101,20110105\n")

    # ---- 语义类非法输入 ----
    def test_date_before_obs_start_aborts(self):
        self._assert_rejected("before_obs", HEADER + "1,2008-01-01,2011-01-05\n")

    def test_date_before_obs_start_aborts_even_with_valid_rows(self):
        """越界日期不能被回看期过滤静默丢掉。"""
        self._assert_rejected("before_obs_mixed",
                              HEADER + "1,2008-01-01,2011-01-05\n2,2011-01-01,2011-01-05\n")

    def test_writes_only_cohort_users(self):
        td = self._case("cohort_only")
        inp, out = os.path.join(td, "in.csv"), os.path.join(td, "out.csv")
        self._write(inp, HEADER + "1,2009-12-05,2011-01-05\n2,2011-01-01,2011-01-05\n")
        self.assertEqual(self._run(inp, out), 0)
        self.assertEqual(pd.read_csv(out)["user_id"].tolist(), [2])

    def test_last_before_first_aborts(self):
        self._assert_rejected("bad_order", HEADER + "1,2011-05-01,2011-01-01\n")

    def test_date_after_obs_end_aborts(self):
        self._assert_rejected("after_obs_end", HEADER + "1,2011-01-01,2012-06-01\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
