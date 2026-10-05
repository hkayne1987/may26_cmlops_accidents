"""Tests for src/data/ingest.py, without network access."""

import hashlib

import pytest

from src.data import ingest

CSV = b'"Num_Acc";"jour";"mois"\n"202400000001";"1";"2"\n'
CSV_V2 = b'"Num_Acc";"jour";"mois"\n"202400000001";"1";"3"\n'


@pytest.mark.parametrize(
    "title, expected",
    [
        # Real titles from data.gouv.fr: the naming changes every year.
        ("caracteristiques-2019.csv", ("caracteristiques", 2019)),
        ("carcteristiques-2021.csv", ("caracteristiques", 2021)),
        ("caract-2023.csv", ("caracteristiques", 2023)),
        ("Caract_2024.csv", ("caracteristiques", 2024)),
        ("Lieux_2024.csv", ("lieux", 2024)),
        ("usagers-2022.csv", ("usagers", 2022)),
        ("Vehicules_2024.csv", ("vehicules", 2024)),
        # Not used: before the 2019 format change, or another table.
        ("vehicules_2016.csv", None),
        ("vehicules-immatricule-baac-2024.csv", None),
        ("Description des bases de données", None),
    ],
)
def test_titles_are_mapped_to_tables(title, expected):
    assert ingest.classify(title) == expected


def resource(title, modified="2025-10-21T00:00:00", size=100, checksum=None):
    return {
        "title": title,
        "url": f"https://static.data.gouv.fr/{title}",
        "last_modified": modified,
        "filesize": size,
        "checksum": {"type": "sha1", "value": checksum} if checksum else None,
    }


def test_duplicate_files_keep_the_most_recent():
    files = ingest.parse_resources(
        [
            resource("lieux-2023.csv", modified="2024-10-23T00:00:00"),
            resource("Lieux_2023_corrige.csv", modified="2025-03-01T00:00:00"),
        ]
    )
    assert files["lieux-2023"].title == "Lieux_2023_corrige.csv"


@pytest.fixture
def fake_download(monkeypatch):
    served = {}
    monkeypatch.setattr(ingest, "download", lambda url: served[url])
    return served


def remote_for(title, content, **kwargs):
    files = ingest.parse_resources([resource(title, size=len(content), **kwargs)])
    return files


def test_a_new_year_is_downloaded_and_recorded(tmp_path, fake_download):
    remote = remote_for("Caract_2025.csv", CSV)
    fake_download[remote["caracteristiques-2025"].url] = CSV
    manifest: dict = {}

    changes = ingest.ingest(remote, manifest, raw_dir=tmp_path)

    assert [(c.key, c.reason) for c in changes] == [("caracteristiques-2025", "new")]
    assert (tmp_path / "caracteristiques-2025.csv").read_bytes() == CSV
    assert manifest["caracteristiques-2025"]["sha1"] == hashlib.sha1(CSV).hexdigest()


def test_unchanged_metadata_downloads_nothing(tmp_path, fake_download):
    remote = remote_for("lieux-2023.csv", CSV)
    manifest = {
        "lieux-2023": {
            "url": remote["lieux-2023"].url,
            "last_modified": remote["lieux-2023"].last_modified,
            "filesize": len(CSV),
            "sha1": hashlib.sha1(CSV).hexdigest(),
        }
    }
    # fake_download serves nothing: any download attempt would raise KeyError.
    assert ingest.ingest(remote, manifest, raw_dir=tmp_path) == []


def test_a_corrected_file_is_an_update(tmp_path, fake_download):
    old = {"url": "x", "last_modified": "2023-10-05", "filesize": 1, "sha1": "old"}
    remote = remote_for("usagers-2022.csv", CSV_V2, modified="2025-03-26T00:00:00")
    fake_download[remote["usagers-2022"].url] = CSV_V2

    changes = ingest.ingest(remote, {"usagers-2022": old}, raw_dir=tmp_path)
    assert [(c.key, c.reason) for c in changes] == [("usagers-2022", "updated")]


def test_new_metadata_with_same_content_is_not_a_change(tmp_path, fake_download):
    """A re-upload of identical bytes must not trigger a retrain."""
    sha1 = hashlib.sha1(CSV).hexdigest()
    manifest = {
        "lieux-2024": {"url": "old", "last_modified": "x", "filesize": 1, "sha1": sha1}
    }
    remote = remote_for("Lieux_2024.csv", CSV)
    fake_download[remote["lieux-2024"].url] = CSV

    assert ingest.ingest(remote, manifest, raw_dir=tmp_path) == []
    assert manifest["lieux-2024"]["url"] == remote["lieux-2024"].url
    assert not (tmp_path / "lieux-2024.csv").exists()


def test_a_checksum_mismatch_is_refused(tmp_path, fake_download):
    remote = remote_for("lieux-2025.csv", CSV, checksum="0" * 40)
    fake_download[remote["lieux-2025"].url] = CSV
    with pytest.raises(ValueError, match="checksum"):
        ingest.ingest(remote, {}, raw_dir=tmp_path)


def test_a_file_that_is_not_baac_is_refused(tmp_path, fake_download):
    remote = remote_for("lieux-2025.csv", b"<html>Maintenance</html>")
    fake_download[remote["lieux-2025"].url] = b"<html>Maintenance</html>"
    with pytest.raises(ValueError, match="BAAC"):
        ingest.ingest(remote, {}, raw_dir=tmp_path)


def test_dry_run_downloads_nothing(tmp_path, fake_download):
    remote = remote_for("Caract_2025.csv", CSV)
    changes = ingest.ingest(remote, {}, raw_dir=tmp_path, dry_run=True)
    assert [c.key for c in changes] == ["caracteristiques-2025"]
    assert not list(tmp_path.iterdir())


def test_github_outputs(tmp_path, monkeypatch):
    output = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    ingest.write_github_outputs(
        [ingest.Change("lieux-2025", "new", "Lieux_2025.csv", "2026-10-20T00:00")]
    )
    text = output.read_text()
    assert "changed=true" in text
    assert "lieux-2025" in text and "2026-10-20" in text
