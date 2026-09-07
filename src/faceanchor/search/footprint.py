"""A subject's footprint across a fixed corpus — every appearance, ranked.

`run --by-face` answers one question and then stops: *is there a match above
threshold, and what is the best one?* That is the right shape for anchoring —
a bundle commits to one match — and the wrong shape for understanding a
corpus. The questions a fixed corpus makes answerable are different:

- How many images in it contain this person, not just the top one?
- What does the score *distribution* look like — is the best hit standing
  clear of the pack, or is it 0.37 in a cloud of 0.36s?
- Where does the threshold actually fall for this probe?
- Did a change to the detector, the encoder, or the threshold move any of it?

That last one is why this exists: over a corpus that does not change, the
footprint is a stable artifact you can diff. Over live timelines it is noise.

The scan reuses the pipeline's own cascade rather than reimplementing it —
same pHash short-circuit, same YuNet detector, same SFace batch, same cosine.
A footprint that scored candidates differently from `run` would be measuring
the wrong thing.

Nothing here anchors, and nothing here persists an embedding: the corpus
stores geometry only (`corpus_db.py`), and the embeddings this computes live
in memory for the length of the scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from faceanchor.search.corpus_db import CorpusDb, CorpusDbError, DbEntry
from faceanchor.search.score import (
    BATCH_SIZE,
    COSINE_ACCEPT_THRESHOLD,
    PHASH_ACCEPT_HAMMING,
    ProbeContext,
)
from faceanchor.vision import align, detect, embed, phash
from faceanchor.vision.decode import decode_jpeg_bytes


@dataclass(frozen=True)
class Appearance:
    """One image in the corpus, and how strongly the probe appears in it."""

    sha256: str
    score: float
    metric: str            # "phash" | "cosine"
    platform: str
    post_uri: str
    image_url: str
    author: str | None
    faces_in_image: int
    phash_hamming: int

    @property
    def above_threshold(self) -> bool:
        return self.metric == "phash" or self.score >= COSINE_ACCEPT_THRESHOLD

    def to_json(self) -> dict:
        return {
            "sha256": self.sha256,
            "score": round(self.score, 4),
            "metric": self.metric,
            "above_threshold": self.above_threshold,
            "platform": self.platform,
            "post_uri": self.post_uri,
            "image_url": self.image_url,
            "author": self.author,
            "faces_in_image": self.faces_in_image,
            "phash_hamming": self.phash_hamming,
        }


@dataclass
class Footprint:
    """Every scored image, plus what the scan could not score and why."""

    appearances: list[Appearance] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    corpus_images: int = 0
    corpus_manifest_sha256: str = ""
    threshold: float = COSINE_ACCEPT_THRESHOLD

    @property
    def matches(self) -> list[Appearance]:
        return [a for a in self.appearances if a.above_threshold]

    def top(self, n: int = 10) -> list[Appearance]:
        return self.appearances[:n]

    def distribution(self) -> dict:
        """Where the scores actually sit.

        The headline number of a face search is meaningless without this: a
        best hit of 0.42 means one thing when the runner-up is 0.11 and quite
        another when it is 0.41, and only the second case is ambiguous.
        """
        cosines = [a.score for a in self.appearances if a.metric == "cosine"]
        if not cosines:
            return {"scored": 0}
        arr = np.array(cosines, dtype=np.float64)
        best = float(arr.max())
        runner_up = float(np.sort(arr)[-2]) if arr.size > 1 else None
        return {
            "scored": int(arr.size),
            "max": round(best, 4),
            "runner_up": round(runner_up, 4) if runner_up is not None else None,
            # How far clear the best hit stands. A margin near zero is the
            # signal to go look at the contact sheet before believing it.
            "margin": round(best - runner_up, 4) if runner_up is not None else None,
            "mean": round(float(arr.mean()), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "above_threshold": int((arr >= COSINE_ACCEPT_THRESHOLD).sum()),
        }

    def to_json(self) -> dict:
        return {
            "kind": "faceanchor.footprint",
            "corpus": {"images": self.corpus_images, "manifest_sha256": self.corpus_manifest_sha256},
            "threshold": self.threshold,
            "matches": len(self.matches),
            "distribution": self.distribution(),
            "skipped": self.skipped,
            "appearances": [a.to_json() for a in self.appearances],
        }


def scan(probe: ProbeContext, db: CorpusDb, *, limit: int | None = None) -> Footprint:
    """Score the probe against every image in `db`, best first.

    Unlike `score_candidates`, this does not early-exit on a strong hit: the
    whole point is the full distribution, and stopping at the first 0.50 would
    hide the second one.
    """
    footprint = Footprint(
        corpus_images=len(db.entries),
        corpus_manifest_sha256=db.manifest_sha256() if db.exists() else "",
    )
    skipped: dict[str, int] = {}

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    survivors: list[tuple[DbEntry, np.ndarray, int]] = []  # (entry, aligned crop, hamming)
    entries = sorted(db.entries.values(), key=lambda e: e.sha256)
    if limit is not None:
        entries = entries[:limit]

    for entry in entries:
        try:
            raw = db.read_image(entry)
        except CorpusDbError:
            # A missing file or a hash mismatch — counted and skipped rather
            # than aborting the scan, so one corrupted entry does not cost you
            # the other 191 scores.
            _skip("unreadable")
            continue
        try:
            image = decode_jpeg_bytes(raw)
        except ValueError:
            _skip("undecodable")
            continue

        hamming = phash.hamming_distance(probe.phash, bytes.fromhex(entry.phash)) if entry.phash else 64
        if hamming <= PHASH_ACCEPT_HAMMING:
            footprint.appearances.append(
                Appearance(
                    sha256=entry.sha256, score=1.0, metric="phash", platform=entry.platform,
                    post_uri=entry.post_uri, image_url=entry.image_url, author=entry.author,
                    faces_in_image=entry.face_count, phash_hamming=hamming,
                )
            )
            continue

        faces = detect.detect_faces(image)
        if not faces:
            _skip("no_face_at_scan_time")
            continue
        best_face = max(faces, key=lambda f: f.score)
        survivors.append((entry, align.align_and_crop(image, best_face), hamming))

    for start in range(0, len(survivors), BATCH_SIZE):
        batch = survivors[start : start + BATCH_SIZE]
        embeddings = embed.embed_batch([crop for _, crop, _ in batch])
        for (entry, _, hamming), cand_embed in zip(batch, embeddings):
            footprint.appearances.append(
                Appearance(
                    sha256=entry.sha256,
                    score=float(embed.cosine_similarity(probe.embedding, cand_embed)),
                    metric="cosine",
                    platform=entry.platform,
                    post_uri=entry.post_uri,
                    image_url=entry.image_url,
                    author=entry.author,
                    faces_in_image=entry.face_count,
                    phash_hamming=hamming,
                )
            )

    footprint.appearances.sort(key=lambda a: a.score, reverse=True)
    footprint.skipped = dict(sorted(skipped.items()))
    return footprint


def write_report(footprint: Footprint, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(footprint.to_json(), indent=2))
