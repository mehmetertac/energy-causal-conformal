"""Execute capstone notebooks top-to-bottom (non-interactive smoke run)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "notebooks/01_synthetic_control.ipynb",
    "notebooks/02_conformal_forecast.ipynb",
    "notebooks/03_gp_solar.ipynb",
]


def main() -> int:
    python = sys.executable
    failures: list[str] = []
    for nb_rel in NOTEBOOKS:
        nb_path = REPO_ROOT / nb_rel
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "notebook.py"
            subprocess.run(
                [
                    python,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    str(nb_path),
                    "--output",
                    script_path.stem,
                    f"--output-dir={tmp}",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            scripts = sorted(Path(tmp).glob("*.py")) + sorted(Path(tmp).glob("*.txt"))
            if not scripts:
                msg = f"no script generated for {nb_rel}"
                raise FileNotFoundError(msg)
            generated = scripts[0]
            print(f"Running {nb_rel} ...")
            proc = subprocess.run(
                [python, str(generated)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                failures.append(nb_rel)
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
            else:
                print(f"OK: {nb_rel}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
