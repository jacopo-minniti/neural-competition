#!/usr/bin/env python3
"""Check maintained documentation, source snapshots, and optional local evidence."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-data",
        action="store_true",
        help="Also verify every file in the ignored local retention inventory",
    )
    args = parser.parse_args()
    errors, links = [], 0
    documents = [ROOT / "README.md", ROOT / "OBJECTIVE.md"]
    for folder in ["docs", "src", "scripts", "tests", "configs"]:
        documents.extend((ROOT / folder).rglob("*.md"))
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
    manifest_count = 0
    for study, folder in [("residual_growth", "source_snapshots"), ("feature_profile", "sources")]:
        snapshots = ROOT / "docs/tables" / study / folder
        for snapshot in snapshots.glob("*.txt"):
            if sha256(snapshot) != snapshot.stem:
                errors.append(f"Invalid source snapshot: {snapshot.name}")
        manifests = json.loads((snapshots.parent / "provenance.json").read_text())
        manifest_count += len(manifests)
        for manifest in manifests:
            for source, expected in manifest["sources"].items():
                path = ROOT / source
                if path.exists() and sha256(path) == expected:
                    continue
                snapshot = snapshots / f"{expected}.txt"
                if not snapshot.exists() or sha256(snapshot) != expected:
                    errors.append(f"Missing executed source: {source} ({expected})")
    local_files = 0
    if args.local_data:
        inventory = json.loads((ROOT / "docs/provenance/local_retention.json").read_text())
        for item in inventory["files"]:
            path = (ROOT / item["path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                errors.append(f"Missing local file: {item['path']}")
            elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                errors.append(f"Changed local file: {item['path']}")
            local_files += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Checked {len(documents)} documents, {links} local links, "
        f"{manifest_count} source manifests, and {local_files} local files."
    )


if __name__ == "__main__":
    main()
