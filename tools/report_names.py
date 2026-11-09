"""Names of the reports in results/: <main factor>_s<N>, where N numbers those reports in chronological order
(s1 = the first). A study keeps its report name in its manifest ("report"), so rebuilding the report overwrites the
same file; a new study gets the next free number.
"""
from __future__ import annotations

import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPORT_NUMBER = re.compile(r"_s(\d+)\.(md|pdf)$")


def next_report_number() -> int:
    numbers = [int(match.group(1)) for path in RESULTS_DIR.glob("*") if (match := REPORT_NUMBER.search(path.name))]
    return max(numbers, default=0) + 1


def report_stem(main_factor: str, manifest: dict) -> str:
    """The manifest's stored report name, or a new "<main_factor>_s<next number>" stored into the manifest
    (the caller saves the manifest)."""
    if not manifest.get("report"):
        manifest["report"] = f"{main_factor}_s{next_report_number()}"
    return manifest["report"]
