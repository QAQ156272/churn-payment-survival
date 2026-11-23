# 环境说明

## 实测环境（2026-11-21）

| 项 | 值 |
|---|---|
| Python | 3.12.14 |
| 解释器 | `.venv/Scripts/python.exe` |
| 依赖管理 | uv 0.13.9（`tools/uv.exe`） |
| pandas | 2.3.3（锁 `<3`，避免与 lifelines 冲突） |
| lifelines | 0.30.3 |
| scipy / numpy | 1.18.1 / 2.5.5 |

## 关键坑：pip 在本机装不了包

`pip install` 会以 `PermissionError` 失败，根因是 DSH 沙箱对临时目录的权限限制：

- `ensurepip` 解压 wheel 到 `TEMP` → 写入/清理被拒（`WinError 5`）
- `pip` 下载包后读临时文件 → `Errno 13`
- `setuptools` 构建 sdist 写临时目录 → 同样被拒

**结论：一律用 `uv` 装包，不要用 pip。**

```bash
# 装新包
export UV_CACHE_DIR="$(cygpath -w "$PWD/.tmp/uvcache")"
./tools/uv.exe pip install --python ./.venv/Scripts/python.exe --find-links vendor/wheels <包名>
```

## `vendor/wheels/` 是什么

`autograd-gamma==0.5.0` 在 PyPI 上只有 sdist 没有 wheel，构建时会撞上面同样的临时目录权限墙。
所以这个 wheel 是从 sdist 手工重打包的（纯 Python，仅 `__init__.py`）。

**删了它就得重新手工构建**，务必保留。

## 关键坑：`tempfile.TemporaryDirectory()` 建出来的目录不可写

本机沙箱下，`tempfile.TemporaryDirectory()` 创建的目录**无法写入文件**：

```
PermissionError: [Errno 13] Permission denied: '...\tmpXXXX\in.csv'
```

把 `tempfile.tempdir` 指向工作区内的 `.tmp/` 也没用——问题出在 `TemporaryDirectory`
建目录时设置的权限，不在路径。

**结论：脚本和测试里不要用 `TemporaryDirectory`，改用 `os.makedirs` 自己建目录。**

`tests/test_build_survival.py` 已按这个方式写：

```python
_TMP_ROOT = os.path.join(_PROJECT_ROOT, ".tmp", "test-tmp")
os.makedirs(_TMP_ROOT, exist_ok=True)
```

## 验证方式

```bash
./.venv/Scripts/python.exe -c "import lifelines, pandas; print(lifelines.__version__, pandas.__version__)"
```

已实测跑通：KaplanMeierFitter、CoxPHFitter（c-index 0.6561）、proportional_hazard_test。
