"""Detects new or corrected BAAC files on data.gouv.fr and downloads them.

The BAAC dataset gets a new year each autumn, and past years are sometimes
corrected (usagers-2022 was republished in 2025). This module compares what
data.gouv.fr publishes with data/sources.json, a manifest of the files the
project currently uses, and downloads what changed into data/raw/.

What it does not do: version the data or retrain. The GitHub workflow that
runs it (.github/workflows/ingest.yml) pushes the new files with DVC and opens
a pull request, so a human reviews the change before retraining starts.

Two quirks of the data.gouv.fr API shape the logic:
- File names change from year to year (caracteristiques-2019,
  carcteristiques-2021, caract-2023, Caract_2024), so tables are recognised
  by prefix, and files are saved under one naming scheme.
- The API omits the checksum for some files (all of 2024). A file is
  therefore considered changed when its date or size moves, then downloaded
  and hashed here: only a different content counts as a change.

Run: python -m src.data.ingest [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.preprocess import FIRST_YEAR, RAW_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# The dataset id is stable; its title and slug name the covered years and
# change with each release.
DATASET_ID = "53698f4ca3a729239d2036df"
API_URL = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"
MANIFEST_PATH = Path("data/sources.json")

# Title prefixes seen on data.gouv.fr, mapped to our table names.
TABLE_PREFIXES = {
    "caract": "caracteristiques",
    "carct": "caracteristiques",  # typo in the 2021 and 2022 titles
    "lieux": "lieux",
    "usager": "usagers",
    "vehicul": "vehicules",
}
# Same years, but a different table (registered vehicles), not used here.
IGNORED_MARKERS = ("immatricul",)


@dataclass
class RemoteFile:
    table: str
    year: int
    title: str
    url: str
    last_modified: str
    filesize: int | None
    checksum: str | None  # sha1 announced by the API, often missing

    @property
    def key(self) -> str:
        return f"{self.table}-{self.year}"


@dataclass
class Change:
    key: str
    reason: str  # "new" or "updated"
    title: str
    last_modified: str


def classify(title: str) -> tuple[str, int] | None:
    """Maps a data.gouv.fr title to (table, year), or None if not a BAAC table."""
    lowered = title.lower()
    if any(marker in lowered for marker in IGNORED_MARKERS):
        return None
    match = re.search(r"(20\d\d)", lowered)
    if not match or int(match.group(1)) < FIRST_YEAR:
        return None
    table = next((t for p, t in TABLE_PREFIXES.items() if lowered.startswith(p)), None)
    return (table, int(match.group(1))) if table else None


def parse_resources(resources: list[dict[str, Any]]) -> dict[str, RemoteFile]:
    """Keeps the BAAC tables from the API listing, one file per table and year."""
    files: dict[str, RemoteFile] = {}
    for resource in resources:
        found = classify(resource.get("title", ""))
        if found is None:
            continue
        checksum = (resource.get("checksum") or {}).get("value")
        remote = RemoteFile(
            table=found[0],
            year=found[1],
            title=resource["title"],
            url=resource["url"],
            last_modified=resource["last_modified"],
            filesize=resource.get("filesize"),
            checksum=checksum,
        )
        previous = files.get(remote.key)
        if previous is not None:
            log.warning(
                f"Two files for {remote.key} ({previous.title!r}, {remote.title!r}): "
                "keeping the most recent"
            )
            if previous.last_modified >= remote.last_modified:
                continue
        files[remote.key] = remote
    return files


def fetch_remote_files() -> dict[str, RemoteFile]:
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        dataset = json.load(response)
    return parse_resources(dataset["resources"])


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(manifest: dict[str, dict[str, Any]], path: Path = MANIFEST_PATH):
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")


def needs_check(remote: RemoteFile, known: dict[str, Any] | None) -> bool:
    """Whether the announced metadata differ from what the manifest recorded."""
    if known is None:
        return True
    return (
        remote.url != known.get("url")
        or remote.last_modified != known.get("last_modified")
        or remote.filesize != known.get("filesize")
    )


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def looks_like_baac_csv(content: bytes) -> bool:
    """Cheap sanity check: a ';'-separated header with an accident id."""
    header = content[:2000].split(b"\n", 1)[0]
    return b";" in header and (b"Num_Acc" in header or b"Accident_Id" in header)


def ingest(
    remote_files: dict[str, RemoteFile],
    manifest: dict[str, dict[str, Any]],
    raw_dir: Path = RAW_DIR,
    dry_run: bool = False,
) -> list[Change]:
    """Downloads new or changed files into raw_dir and updates the manifest.

    The manifest is updated in place. A file whose metadata moved but whose
    content is identical only refreshes the manifest, it is not a change.
    """
    changes: list[Change] = []
    for key, remote in sorted(remote_files.items()):
        known = manifest.get(key)
        if not needs_check(remote, known):
            continue

        if dry_run:
            reason = "new" if known is None else "metadata changed"
            log.info(f"{key}: {reason} ({remote.title}, {remote.last_modified})")
            changes.append(Change(key, reason, remote.title, remote.last_modified))
            continue

        content = download(remote.url)
        sha1 = hashlib.sha1(content).hexdigest()
        if remote.checksum and remote.checksum != sha1:
            raise ValueError(
                f"{key}: downloaded content does not match the checksum "
                f"announced by data.gouv.fr ({remote.checksum})"
            )
        if not looks_like_baac_csv(content):
            raise ValueError(f"{key}: {remote.title} does not look like a BAAC CSV")

        entry = {
            "title": remote.title,
            "url": remote.url,
            "last_modified": remote.last_modified,
            "filesize": remote.filesize,
            "sha1": sha1,
        }
        if known is not None and known.get("sha1") == sha1:
            log.info(f"{key}: metadata changed, content identical")
            manifest[key] = entry
            continue

        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{key}.csv").write_bytes(content)
        manifest[key] = entry
        reason = "new" if known is None else "updated"
        log.info(f"{key}: {reason}, saved ({len(content)} bytes)")
        changes.append(Change(key, reason, remote.title, remote.last_modified))
    return changes


def write_github_outputs(changes: list[Change]) -> None:
    """Hands the result to the next steps of the GitHub workflow."""
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    summary = "\n".join(
        f"- `{c.key}`: {c.reason} ({c.title}, published {c.last_modified[:10]})"
        for c in changes
    )
    with open(output, "a") as f:
        f.write(f"changed={'true' if changes else 'false'}\n")
        f.write(f"summary<<EOF\n{summary}\nEOF\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="report what changed, download nothing"
    )
    args = parser.parse_args()

    remote_files = fetch_remote_files()
    years = sorted({f.year for f in remote_files.values()})
    log.info(f"data.gouv.fr publishes {len(remote_files)} BAAC files for {years}")

    manifest = load_manifest()
    changes = ingest(remote_files, manifest, dry_run=args.dry_run)
    if not args.dry_run:
        save_manifest(manifest)

    if changes:
        log.info(f"{len(changes)} change(s): {[asdict(c)['key'] for c in changes]}")
    else:
        log.info("Data is up to date with data.gouv.fr")
    write_github_outputs(changes)


if __name__ == "__main__":
    main()
