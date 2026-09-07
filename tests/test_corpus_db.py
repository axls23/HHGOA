"""The local corpus database and the footprint scan over it.

Two properties carry most of the weight here and are asserted directly:

1. **No embeddings are ever written to disk.** PRD §3's "no persistent
   embedding store" is what separates this repo from the untargeted
   facial-recognition database EU AI Act Art. 5(1)(e) prohibits, and a corpus
   directory is exactly where that rule would quietly be broken.
2. **Content addressing is checked, not just claimed.** A store that names
   files by their hash and never verifies it has the costs of content
   addressing and none of the benefits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from faceanchor.search.corpus_db import (
    MANIFEST_KIND,
    BuildStats,
    CorpusDb,
    CorpusDbError,
)
from faceanchor.search.models import Candidate

FIXTURE = Path(__file__).parent / "fixtures" / "probe_sample.jpg"


def _candidate(url: str = "https://files.mastodon.social/a.jpg", platform: str = "mastodon") -> Candidate:
    return Candidate(
        platform=platform, image_url=url, post_uri="https://mastodon.social/@a/1",
        author="a", extra={"variant": "preview"},
    )


def _built(tmp_path: Path, n: int = 1) -> CorpusDb:
    db = CorpusDb(tmp_path / "db")
    raw = FIXTURE.read_bytes()
    stats = BuildStats()
    db.add(_candidate(), raw, corpus="mastodon:tag/selfie", stats=stats)
    for i in range(1, n):
        # distinct bytes per entry, so content addressing has something to do
        db.add(_candidate(f"https://x/{i}.jpg"), raw + b"\x00" * i, corpus="mastodon:tag/selfie", stats=stats)
    db.save(corpora=["mastodon:tag/selfie"])
    return db


# ── privacy: the rule the whole store is built around ────────────────────


def _float_runs(node) -> list[list]:
    """Every list of numbers anywhere in a JSON structure.

    Structural rather than string-matching: a 128-d template would be a long
    run of floats under *some* key, and renaming the key would not help it
    escape this.
    """
    runs = []
    if isinstance(node, dict):
        for value in node.values():
            runs += _float_runs(value)
    elif isinstance(node, list):
        if node and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node):
            runs.append(node)
        else:
            for value in node:
                runs += _float_runs(value)
    return runs


def test_no_embeddings_are_persisted_anywhere(tmp_path):
    db = _built(tmp_path, n=3)
    manifest = (tmp_path / "db" / "manifest.jsonl").read_text()

    for line in manifest.splitlines():
        row = json.loads(line)
        # No key anywhere may hold a vector. SFace emits 128 floats; a bbox is
        # 4, so anything longer than a handful is the thing we forbid.
        for run in _float_runs(row):
            assert len(run) <= 4, f"a {len(run)}-element numeric vector reached disk: {run[:8]}…"
        assert "embedding" not in {k.lower() for k in row}
        assert "vector" not in {k.lower() for k in row}

    entry = next(iter(db.entries.values()))
    assert "embedding" not in entry.to_json()
    # face geometry is stored, and geometry is not a biometric template
    assert set(entry.faces[0]) == {"bbox", "score"}
    assert len(entry.faces[0]["bbox"]) == 4

    # and nothing else in the directory is a stray dump
    for path in (tmp_path / "db").rglob("*"):
        if path.is_file() and path.suffix in {".npy", ".npz", ".bin", ".pkl", ".pt"}:
            raise AssertionError(f"unexpected binary artifact in the corpus store: {path}")


def test_manifest_states_that_it_holds_no_embeddings(tmp_path):
    """A reader of the file should not have to take a docstring's word for it."""
    _built(tmp_path)
    header = json.loads((tmp_path / "db" / "manifest.jsonl").read_text().splitlines()[0])
    assert header["kind"] == MANIFEST_KIND
    assert header["contains_embeddings"] is False
    assert "no persistent embedding store" in header["note"]


# ── content addressing ───────────────────────────────────────────────────


def test_images_are_stored_under_their_own_sha256(tmp_path):
    db = _built(tmp_path)
    entry = next(iter(db.entries.values()))
    path = db.image_path(entry)
    assert path.exists()
    assert path.name == entry.sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry.sha256


def test_reading_verifies_the_hash_rather_than_trusting_the_name(tmp_path):
    db = _built(tmp_path)
    entry = next(iter(db.entries.values()))
    db.image_path(entry).write_bytes(b"swapped out from under it")
    with pytest.raises(CorpusDbError, match="does not match its name"):
        db.read_image(entry)


def test_missing_image_file_is_an_error_not_a_silent_skip(tmp_path):
    db = _built(tmp_path)
    entry = next(iter(db.entries.values()))
    db.image_path(entry).unlink()
    with pytest.raises(CorpusDbError, match="missing"):
        db.read_image(entry)


def test_deduplication_is_by_content_not_url(tmp_path):
    """The same photo reposted under two URLs is one entry — a URL-keyed store
    would double-count it and then score it twice."""
    db = CorpusDb(tmp_path / "db")
    raw = FIXTURE.read_bytes()
    stats = BuildStats()
    assert db.add(_candidate("https://a/1.jpg"), raw, stats=stats) is not None
    assert db.add(_candidate("https://b/2.jpg"), raw, stats=stats) is None
    assert len(db.entries) == 1
    assert stats.added == 1 and stats.duplicate == 1


# ── build-time rejections ────────────────────────────────────────────────


def test_undecodable_and_faceless_images_are_rejected(tmp_path):
    db = CorpusDb(tmp_path / "db")
    stats = BuildStats()
    assert db.add(_candidate("https://x/bad.jpg"), b"not an image", stats=stats) is None
    assert stats.undecodable == 1

    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2

    ok, buf = cv2.imencode(".jpg", blank)
    assert ok
    assert db.add(_candidate("https://x/blank.jpg"), buf.tobytes(), stats=stats) is None
    assert stats.no_face == 1
    assert db.entries == {}


# ── round trip ───────────────────────────────────────────────────────────


def test_save_load_round_trip_preserves_provenance(tmp_path):
    original = _built(tmp_path, n=3)
    reloaded = CorpusDb(tmp_path / "db").load()

    assert set(reloaded.entries) == set(original.entries)
    entry = reloaded.entries[next(iter(original.entries))]
    assert entry.platform == "mastodon"
    assert entry.post_uri == "https://mastodon.social/@a/1"
    assert entry.author == "a"
    assert entry.corpus == "mastodon:tag/selfie"
    assert entry.variant == "preview"
    assert entry.phash and entry.width and entry.height


def test_as_fetched_matches_what_the_scoring_cascade_consumes(tmp_path):
    """The replay must be indistinguishable in shape from a live fetch, so the
    same `score_candidates` runs over both."""
    from faceanchor.search import score
    from faceanchor.vision import detect, quality
    from faceanchor.vision.decode import decode_jpeg_bytes

    db = _built(tmp_path, n=3)
    fetched = db.as_fetched()
    assert len(fetched) == 3
    assert all(isinstance(c, Candidate) and isinstance(b, bytes) for c, b in fetched)
    # provenance survives the round trip — a bundle must point at the real post
    assert all(c.post_uri.startswith("https://") for c, _ in fetched)
    assert all(c.extra["source"] == "corpus_db" for c, _ in fetched)

    image = decode_jpeg_bytes(FIXTURE.read_bytes())
    face, err = quality.check_single_face(detect.detect_faces(image), None)
    assert err is None
    matches = score.score_candidates(score.build_probe_context(image, face), fetched)
    assert matches, "the probe's own image is in the corpus and must be retrieved"


def test_load_rejects_a_foreign_or_broken_manifest(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    (root / "manifest.jsonl").write_text('{"kind":"something.else"}\n')
    with pytest.raises(CorpusDbError, match="not a faceanchor.corpus_db manifest"):
        CorpusDb(root).load()

    (root / "manifest.jsonl").write_text("not json at all\n")
    with pytest.raises(CorpusDbError, match="header is not valid JSON"):
        CorpusDb(root).load()

    with pytest.raises(CorpusDbError, match="no corpus database"):
        CorpusDb(tmp_path / "nope").load()


def test_manifest_sha256_pins_the_corpus(tmp_path):
    """The bundle records this, so it must change when the corpus does."""
    db = _built(tmp_path, n=2)
    before = db.manifest_sha256()
    db.add(_candidate("https://x/new.jpg"), FIXTURE.read_bytes() + b"\x01\x02")
    db.save(corpora=["mastodon:tag/selfie"])
    assert db.manifest_sha256() != before


def test_stats_summarise_the_corpus(tmp_path):
    db = _built(tmp_path, n=3)
    stats = db.stats()
    assert stats["images"] == 3
    assert stats["faces"] >= 3
    assert stats["by_platform"] == {"mastodon": 3}
    assert stats["bytes"] > 0


# ── footprint ────────────────────────────────────────────────────────────


def _probe_context():
    from faceanchor.search import score
    from faceanchor.vision import detect, quality
    from faceanchor.vision.decode import decode_jpeg_bytes

    image = decode_jpeg_bytes(FIXTURE.read_bytes())
    face, err = quality.check_single_face(detect.detect_faces(image), None)
    assert err is None
    return score.build_probe_context(image, face)


def test_footprint_finds_the_probe_and_ranks_everything(tmp_path):
    from faceanchor.search import footprint as footprint_mod

    db = _built(tmp_path, n=3)
    result = footprint_mod.scan(_probe_context(), db)

    assert result.corpus_images == 3
    assert len(result.matches) >= 1
    assert result.appearances == sorted(result.appearances, key=lambda a: a.score, reverse=True)
    assert result.corpus_manifest_sha256 == db.manifest_sha256()
    assert result.matches[0].post_uri.startswith("https://")


def test_footprint_reports_the_distribution_not_just_the_winner(tmp_path):
    """A best hit of 0.42 means one thing when the runner-up is 0.11 and quite
    another when it is 0.41 — only the second is ambiguous."""
    from faceanchor.search.footprint import Appearance, Footprint

    fp = Footprint(
        appearances=[
            Appearance("a", 0.42, "cosine", "bsky", "u", "i", None, 1, 30),
            Appearance("b", 0.11, "cosine", "bsky", "u", "i", None, 1, 30),
            Appearance("c", 0.09, "cosine", "bsky", "u", "i", None, 1, 30),
        ]
    )
    dist = fp.distribution()
    assert dist["scored"] == 3
    assert dist["max"] == 0.42
    assert dist["runner_up"] == 0.11
    assert dist["margin"] == pytest.approx(0.31, abs=1e-6)
    assert dist["above_threshold"] == 1
    assert len(fp.matches) == 1

    assert Footprint().distribution() == {"scored": 0}


def test_footprint_does_not_early_exit_on_a_strong_hit(tmp_path):
    """`score_candidates` stops at the first 0.50 because it only needs one
    match. A footprint that did the same would hide the second one."""
    from faceanchor.search import footprint as footprint_mod

    db = _built(tmp_path, n=4)
    result = footprint_mod.scan(_probe_context(), db)
    assert len(result.appearances) == 4


def test_footprint_report_is_self_describing(tmp_path):
    from faceanchor.search import footprint as footprint_mod

    db = _built(tmp_path, n=2)
    result = footprint_mod.scan(_probe_context(), db)
    out = tmp_path / "footprint.json"
    footprint_mod.write_report(result, out)

    data = json.loads(out.read_text())
    assert data["kind"] == "faceanchor.footprint"
    assert data["corpus"]["manifest_sha256"] == db.manifest_sha256()
    assert data["threshold"] == 0.363
    assert len(data["appearances"]) == 2
    assert "embedding" not in out.read_text().lower()


def test_footprint_limit_bounds_the_scan(tmp_path):
    from faceanchor.search import footprint as footprint_mod

    db = _built(tmp_path, n=4)
    assert len(footprint_mod.scan(_probe_context(), db, limit=2).appearances) == 2


# ── CLI plumbing ─────────────────────────────────────────────────────────


def test_run_rejects_corpus_db_outside_by_face():
    from typer.testing import CliRunner

    from faceanchor.cli import app

    result = CliRunner().invoke(
        app,
        ["run", "--probe", str(FIXTURE), "--query", "x", "--corpus-db", "/tmp/nope",
         "--chain", "anvil", "--contract-address", "0x0"],
    )
    assert result.exit_code != 0
    assert "only applies to --by-face" in result.output


def test_by_face_accepts_a_corpus_db_instead_of_a_spec():
    """--by-face used to require --corpus; a database is now an alternative."""
    from typer.testing import CliRunner

    from faceanchor.cli import app

    result = CliRunner().invoke(
        app, ["run", "--probe", str(FIXTURE), "--by-face", "--chain", "anvil", "--contract-address", "0x0"]
    )
    assert "--corpus-db <dir>" in result.output


def test_corpus_footprint_enforces_the_consent_gate(tmp_path):
    from typer.testing import CliRunner

    from faceanchor.cli import app

    _built(tmp_path)
    result = CliRunner().invoke(
        app, ["corpus-footprint", "--probe", str(FIXTURE), "--db", str(tmp_path / "db")]
    )
    assert result.exit_code != 0
    assert "subject-consent is required" in result.output


def test_corpus_build_requires_a_source():
    from typer.testing import CliRunner

    from faceanchor.cli import app

    result = CliRunner().invoke(app, ["corpus-build", "--db", "/tmp/x"])
    assert result.exit_code != 0
    assert "--corpus" in result.output and "--from-dir" in result.output
