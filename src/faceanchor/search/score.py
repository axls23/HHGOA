"""Scoring cascade, cheapest filter first (PRD §5.2):

1. pHash Hamming <= 8 -> immediate high-confidence match, skip embedding.
2. Else YuNet detect on the candidate; no face -> drop.
3. Batch embed survivors (CUDA EP, batch_size up to 32).
4. cosine >= 0.363 accept; early-exit the whole fanout at cosine >= 0.50.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faceanchor.search.fanout import sha256_hex
from faceanchor.search.models import Candidate, ScoredMatch
from faceanchor.vision import align, detect, embed, phash
from faceanchor.vision.decode import decode_jpeg_bytes

PHASH_ACCEPT_HAMMING = 8
COSINE_ACCEPT_THRESHOLD = 0.363
COSINE_EARLY_EXIT_THRESHOLD = 0.50
BATCH_SIZE = 32


@dataclass(frozen=True)
class ProbeContext:
    embedding: np.ndarray   # unit-norm 128-d
    phash: bytes


def build_probe_context(probe_image_bgr, probe_face) -> ProbeContext:
    crop = align.align_and_crop(probe_image_bgr, probe_face)
    return ProbeContext(embedding=embed.embed_one(crop), phash=phash.phash(probe_image_bgr))


def score_candidates(probe: ProbeContext, fetched: list[tuple[Candidate, bytes]]) -> list[ScoredMatch]:
    """Runs the full cascade over already-fetched candidate image bytes.

    Step 1 (pHash) alone removes ~70% of embedding calls on real corpora
    (PRD §5.2) — it runs on the whole candidate image, not a detected face
    crop, since it's meant to catch exact/near-exact re-uploads of the probe.
    """
    matches: list[ScoredMatch] = []
    survivors: list[tuple[Candidate, bytes, np.ndarray]] = []  # (candidate, raw_bytes, aligned_crop)

    for candidate, raw_bytes in fetched:
        try:
            image = decode_jpeg_bytes(raw_bytes)
        except ValueError:
            continue

        cand_hash = phash.phash(image)
        hamming = phash.hamming_distance(probe.phash, cand_hash)
        if hamming <= PHASH_ACCEPT_HAMMING:
            matches.append(
                ScoredMatch(candidate=candidate, score=1.0, metric="phash", image_bytes_sha256=sha256_hex(raw_bytes))
            )
            if hamming == 0:
                return matches  # exact re-upload — nothing else will beat this
            continue

        faces = detect.detect_faces(image)
        if not faces:
            continue
        best_face = max(faces, key=lambda f: f.score)
        crop = align.align_and_crop(image, best_face)
        survivors.append((candidate, raw_bytes, crop))

    for batch_start in range(0, len(survivors), BATCH_SIZE):
        batch = survivors[batch_start : batch_start + BATCH_SIZE]
        embeddings = embed.embed_batch([crop for _, _, crop in batch])
        for (candidate, raw_bytes, _), cand_embed in zip(batch, embeddings):
            cosine = embed.cosine_similarity(probe.embedding, cand_embed)
            if cosine < COSINE_ACCEPT_THRESHOLD:
                continue
            matches.append(
                ScoredMatch(candidate=candidate, score=cosine, metric="cosine", image_bytes_sha256=sha256_hex(raw_bytes))
            )
            if cosine >= COSINE_EARLY_EXIT_THRESHOLD:
                return sorted(matches, key=lambda m: m.score, reverse=True)

    return sorted(matches, key=lambda m: m.score, reverse=True)
