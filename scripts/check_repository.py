#!/usr/bin/env python3
"""Validate maintained documentation links and required evidence records."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    documents = [ROOT / "README.md", ROOT / "OBJECTIVES.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    documents.extend((ROOT / "scripts").rglob("*.md"))
    documents.extend((ROOT / "tests").rglob("*.md"))
    documents.extend((ROOT / "archive").rglob("*.md"))
    documents.extend((ROOT / "figures").rglob("*.md"))

    errors = []
    links = 0
    for document in documents:
        content = re.sub(r"```.*?```", "", document.read_text(), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            links += 1
            path = (document.parent / unquote(parsed.path)).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing link {target}")

    required = [
        "report.tex",
        "figures/learning_and_reserve.png",
        "docs/evidence/competition_reserve/summary.json",
        "docs/evidence/competition_reserve/learning_curve_example_20261030/forecast.json",
        "docs/evidence/reserve_bridge/analysis.json",
        "docs/evidence/natural_parity/analysis.json",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required evidence: {relative}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(documents)} documents, {links} local links, and {len(required)} evidence records.")


if __name__ == "__main__":
    main()
