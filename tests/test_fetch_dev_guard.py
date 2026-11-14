"""experiments/medmcqa/fetch_dev.py: the expected manifest hash is pinned and the import guard speaks."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/medmcqa/fetch_dev.py"
EXPECTED = "5b2ce9718e7cf421d88dd516d9c52b8b282cd886d13cbbc61ef5d7463a90a30d"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_dev", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expected_manifest_hash_is_the_one_the_caches_replay_against():
    assert _load().EXPECTED == EXPECTED


def test_missing_parquet_stack_exits_with_the_install_hint(tmp_path):
    # a sitecustomize that hides pandas and pyarrow, so the guard path runs on any machine
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'pandas', 'pyarrow'}:\n"
        "            raise ImportError(f'No module named {name!r}')\n"
        "sys.meta_path.insert(0, _Block())\n")
    env = {"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path / "x.csv")],
                         check=False, capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode != 0
    assert ".[manifest]" in out.stderr and "pyarrow" in out.stderr
