from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    _run([sys.executable, "scripts/download_data.py"])
    _run([sys.executable, "scripts/build_duckdb.py"])
    _run([sys.executable, "-m", "src.index"])


if __name__ == "__main__":
    main()
