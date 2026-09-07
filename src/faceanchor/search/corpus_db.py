"""A local, content-addressed image corpus — the pipeline's test bench.

Every run of `--by-face` against live timelines is a different experiment:
the tags move, the feed reorders, a source 403s, and a candidate that scored
0.46 yesterday is gone today. That is fine for a demo and useless for
testing, where the question is "did *my change* alter the outcome" and the
answer must not depend on what strangers posted this morning.

So this compiles one: fetch a bounded corpus once, keep the bytes, and let
every subsequent run replay it. Determinism is the point, and three other
things fall out of it:

- **Nothing explicit is re-downloaded.** The safety filter (`search/safety.py`)
  runs at build time, so an explicit post is excluded once rather than on
  every run — and the images that remain on disk are ones that passed it.
- **The network stops mattering.** `run --by-face --corpus-db <dir>` is
  offline, so CI and a demo on a bad connection behave identically.
- **A footprint becomes measurable.** With a fixed corpus, "where does this
  face appear in it" has a stable answer you can diff across changes to the
  detector, the encoder, or the threshold.

## What is stored, and what is deliberately not

Stored: the image bytes (content-addressed by sha256), their pHash, their
source provenance (platform, post URI, redacted image URL, author handle),
and per-image face *geometry* — how many faces, their boxes, detector
confidence.

**Not stored: face embeddings.** PRD §3's "no persistent embedding store" is
what separates this repo from the untargeted facial-recognition database EU
AI Act Art. 5(1)(e) prohibits, and a directory of 128-d templates on disk is
exactly that database. Embeddings are recomputed in memory at query time —
the cost is a batch SFace pass over a few hundred crops, which is seconds,
and the property is worth far more than the seconds.

The corpus is still a pile of strangers' faces on your disk. It is written
under a path you name, `evidence/` is gitignored, and it is yours to delete.
Treat it the way `--contact-sheet` is treated: opt-in, local, never anchored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from faceanchor.search.models import Candidate
from faceanchor.vision import detect
from faceanchor.vision.decode import decode_jpeg_bytes
from faceanchor.vision.phash import phash

MANIFEST_NAME = "manifest.jsonl"
IMAGES_DIRNAME = "images"
MANIFEST_KIND = "faceanchor.corpus_db"
SCHEMA_VERSION = 1


class CorpusDbError(Exception):
    pass


@dataclass(frozen=True)
class DbEntry:
    """One image in the corpus, with everything except its embedding."""

    sha256: str
    phash: str
    bytes_len: int
    width: int
    height: int
    platform: str
    post_uri: str
    image_url: str
    author: str | None
    variant: str
    faces: list[dict] = field(default_factory=list)
    corpus: str = ""
    ingested_at: str = ""

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def relative_path(self) -> str:
        return f"{IMAGES_DIRNAME}/{self.sha256[:2]}/{self.sha256}"

    def to_json(self) -> dict:
        return {
            "sha256": self.sha256,
            "phash": self.phash,
            "bytes": self.bytes_len,
            "width": self.width,
            "height": self.height,
            "platform": self.platform,
            "post_uri": self.post_uri,
            "image_url": self.image_url,
            "author": self.author,
            "variant": self.variant,
            "faces": self.faces,
            "corpus": self.corpus,
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_json(cls, row: dict) -> DbEntry:
        return cls(
            sha256=row["sha256"],
            phash=row.get("phash", ""),
            bytes_len=row.get("bytes", 0),
            width=row.get("width", 0),
            height=row.get("height", 0),
            platform=row.get("platform", ""),
            post_uri=row.get("post_uri", ""),
            image_url=row.get("image_url", ""),
            author=row.get("author"),
            variant=row.get("variant", "original"),
            faces=row.get("faces", []),
            corpus=row.get("corpus", ""),
            ingested_at=row.get("ingested_at", ""),
        )


def _face_geometry(face: detect.Face) -> dict:
    """Box + detector confidence. Geometry, never a template.

    A bounding box says *a face is here*; an embedding says *this is who it
    is*. Only the first is safe to leave on disk (see the module docstring).
    """
    x, y, w, h = (float(v) for v in np.asarray(face.bbox).reshape(-1)[:4])
    return {"bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)], "score": round(float(face.score), 4)}


@dataclass
class BuildStats:
    added: int = 0
    duplicate: int = 0
    undecodable: int = 0
    no_face: int = 0
    already_present: int = 0

    def describe(self) -> str:
        return (
            f"added {self.added}, already present {self.already_present}, "
            f"duplicate bytes {self.duplicate}, undecodable {self.undecodable}, no face {self.no_face}"
        )


class CorpusDb:
    """Directory-backed store: `manifest.jsonl` + content-addressed `images/`.

    JSONL rather than a database file for the same reason the candidate log is
    JSONL — it is greppable, diffable, appendable, and survives being opened
    by anything. There is no index to corrupt and no schema migration to run.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.entries: dict[str, DbEntry] = {}
        self.header: dict = {}

    # ── load / save ──────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load(self) -> CorpusDb:
        if not self.exists():
            raise CorpusDbError(
                f"no corpus database at {self.root} (expected {MANIFEST_NAME}). "
                "Build one first: faceanchor corpus-build --db <dir> --corpus <spec>"
            )
        lines = self.manifest_path.read_text().splitlines()
        if not lines:
            raise CorpusDbError(f"{self.manifest_path} is empty")
        try:
            self.header = json.loads(lines[0])
        except ValueError as e:
            raise CorpusDbError(f"{self.manifest_path} header is not valid JSON: {e}") from e
        if self.header.get("kind") != MANIFEST_KIND:
            raise CorpusDbError(f"{self.manifest_path} is not a {MANIFEST_KIND} manifest")
        for lineno, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            try:
                entry = DbEntry.from_json(json.loads(line))
            except (ValueError, KeyError) as e:
                raise CorpusDbError(f"{self.manifest_path}:{lineno} is malformed: {e}") from e
            self.entries[entry.sha256] = entry
        return self

    def save(self, *, corpora: list[str], safety: dict | None = None, source: str = "live") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Appending a local import to a live-built corpus makes it both, and
        # the header should say so rather than claim whichever ran last.
        previous = self.header.get("source")
        if previous and previous != source:
            source = "mixed"
        header = {
            "kind": MANIFEST_KIND,
            "schema": SCHEMA_VERSION,
            "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "corpora": sorted(corpora),
            "images": len(self.entries),
            "images_with_faces": sum(1 for e in self.entries.values() if e.face_count),
            "faces_total": sum(e.face_count for e in self.entries.values()),
            # Stated in the artifact itself, so a reader of the file never has
            # to take the docstring's word for it.
            "contains_embeddings": False,
            "note": "face geometry only; no embeddings are stored (PRD §3: no persistent embedding store)",
        }
        # An append that fetched nothing has no new filter counts of its own;
        # dropping the ones already recorded would silently erase the fact
        # that the corpus was filtered at all.
        carried = safety or self.header.get("filtered_sensitive")
        if carried:
            header["filtered_sensitive"] = carried
        with self.manifest_path.open("w") as fh:
            fh.write(json.dumps(header) + "\n")
            for entry in sorted(self.entries.values(), key=lambda e: e.sha256):
                fh.write(json.dumps(entry.to_json()) + "\n")
        self.header = header

    def manifest_sha256(self) -> str:
        """Identifies the exact corpus a run was scored against."""
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    # ── read ─────────────────────────────────────────────────────────────
    def image_path(self, entry: DbEntry) -> Path:
        return self.root / entry.relative_path()

    def read_image(self, entry: DbEntry) -> bytes:
        path = self.image_path(entry)
        if not path.exists():
            raise CorpusDbError(f"manifest names {entry.sha256} but {path} is missing")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry.sha256:
            # Content addressing is only worth anything if it is checked.
            raise CorpusDbError(f"{path} does not match its name: manifest {entry.sha256}, file {actual}")
        return raw

    def as_fetched(self, with_faces_only: bool = False) -> list[tuple[Candidate, bytes]]:
        """The corpus in the exact shape `score.score_candidates` consumes.

        Provenance is preserved: `image_url`/`post_uri` still name where the
        image came from, so an evidence bundle built from a replayed corpus
        points at the real post rather than at a local path. That the bytes
        were read from disk is recorded separately, in the discovery block.
        """
        out: list[tuple[Candidate, bytes]] = []
        for entry in sorted(self.entries.values(), key=lambda e: e.sha256):
            if with_faces_only and not entry.face_count:
                continue
            out.append(
                (
                    Candidate(
                        platform=entry.platform,
                        image_url=entry.image_url,
                        post_uri=entry.post_uri,
                        author=entry.author,
                        extra={
                            "variant": entry.variant,
                            "corpus": entry.corpus,
                            "local_sha256": entry.sha256,
                            "source": "corpus_db",
                        },
                    ),
                    self.read_image(entry),
                )
            )
        return out

    # ── write ────────────────────────────────────────────────────────────
    def add(self, candidate: Candidate, raw: bytes, *, corpus: str = "", stats: BuildStats | None = None) -> DbEntry | None:
        """Decode, detect, store. Returns None when the image was rejected.

        Deduplication is by **content**, not URL: the same photo reposted under
        two URLs is one entry, which a URL-keyed store would double-count and
        then double-score.
        """
        stats = stats if stats is not None else BuildStats()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in self.entries:
            stats.duplicate += 1
            return None
        try:
            image = decode_jpeg_bytes(raw)
        except ValueError:
            stats.undecodable += 1
            return None

        faces = detect.detect_faces(image)
        h, w = image.shape[:2]
        entry = DbEntry(
            sha256=digest,
            phash=phash(image).hex(),
            bytes_len=len(raw),
            width=int(w),
            height=int(h),
            platform=candidate.platform,
            post_uri=candidate.post_uri,
            image_url=candidate.image_url,
            author=candidate.author,
            variant=candidate.extra.get("variant", "original"),
            faces=[_face_geometry(f) for f in faces],
            corpus=corpus,
            ingested_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if not entry.faces:
            # Kept out rather than kept: a faceless image can never match a
            # face, so storing it costs disk and scoring time for nothing.
            stats.no_face += 1
            return None

        path = self.image_path(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.entries[digest] = entry
        stats.added += 1
        return entry

    def stats(self) -> dict:
        by_platform: dict[str, int] = {}
        by_corpus: dict[str, int] = {}
        faces = 0
        multi_face = 0
        for entry in self.entries.values():
            by_platform[entry.platform] = by_platform.get(entry.platform, 0) + 1
            if entry.corpus:
                by_corpus[entry.corpus] = by_corpus.get(entry.corpus, 0) + 1
            faces += entry.face_count
            if entry.face_count > 1:
                multi_face += 1
        return {
            "images": len(self.entries),
            "faces": faces,
            "multi_face_images": multi_face,
            "by_platform": dict(sorted(by_platform.items())),
            "by_corpus": dict(sorted(by_corpus.items())),
            "bytes": sum(e.bytes_len for e in self.entries.values()),
        }
