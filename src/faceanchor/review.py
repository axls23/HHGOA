"""Development contact sheet — the candidate links with the images embedded.

A log of URLs and cosine scores tells you *that* something scored 0.46. It
cannot tell you whether 0.46 was the right answer, and at a threshold of 0.363
that is the only question worth asking. So this renders the probe and every
candidate the run actually fetched, thumbnailed and inlined as data URIs, in
score order, with the accept threshold drawn where it falls.

Inlined rather than linked on purpose: social CDN URLs expire (and Meta's are
signed), so a sheet of links rots into a page of broken images exactly when
you go back to review a disputed match.

Never anchored, never part of the evidence bundle (PRD §6.3 excludes image
bytes by construction), and it writes other people's faces to disk — so it is
opt-in, lands under evidence/ which is gitignored, and is yours to delete.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2
import numpy as np

from faceanchor.vision.decode import decode_jpeg_bytes

THUMB_PX = 160
JPEG_QUALITY = 72


def _thumb_data_uri(raw: bytes) -> str | None:
    try:
        image = decode_jpeg_bytes(raw)
    except ValueError:
        return None
    h, w = image.shape[:2]
    if max(h, w) > THUMB_PX:
        scale = THUMB_PX / max(h, w)
        image = cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _cell(thumb: str | None, record: dict, threshold: float) -> str:
    cosine = record.get("cosine")
    verdict = record.get("verdict", "not_scored")
    accepted = cosine is not None and cosine >= threshold
    score_text = f"{cosine:.4f}" if cosine is not None else "—"
    klass = "hit" if accepted else ("scored" if cosine is not None else "unscored")
    img = (
        f'<img src="{thumb}" alt="" loading="lazy">'
        if thumb
        else '<div class="noimg">no decodable image</div>'
    )
    url = html.escape(record.get("post_uri") or record.get("image_url") or "")
    author = html.escape(record.get("author") or "—")
    return f"""<figure class="{klass}">
  {img}
  <figcaption>
    <span class="score">{score_text}</span>
    <span class="verdict">{html.escape(verdict)}</span>
    <span class="author">{author}</span>
    <a href="{url}" rel="noreferrer noopener" target="_blank">post</a>
  </figcaption>
</figure>"""


def write_contact_sheet(
    *,
    path: Path,
    probe_bytes: bytes,
    probe_label: str,
    fetched: list[tuple[object, bytes]],
    score_trace: list[dict],
    threshold: float,
    discovery: dict,
    limit: int = 300,
) -> int:
    """Renders probe + fetched candidates to a single self-contained HTML file.

    Ordering is by cosine descending, so whatever cleared the threshold — or
    came closest to it — is the first thing on screen.
    """
    verdicts = {r["image_url"]: r for r in score_trace}
    rows = []
    for candidate, raw in fetched:
        record = dict(verdicts.get(candidate.image_url, {}))
        record.setdefault("image_url", candidate.image_url)
        record.setdefault("post_uri", candidate.post_uri)
        record.setdefault("author", candidate.author)
        record.setdefault("verdict", "not_scored")
        rows.append((record, raw))

    rows.sort(key=lambda r: (r[0].get("cosine") is None, -(r[0].get("cosine") or 0.0)))
    rows = rows[:limit]

    cells = []
    for record, raw in rows:
        cells.append(_cell(_thumb_data_uri(raw), record, threshold))

    probe_thumb = _thumb_data_uri(probe_bytes) or ""
    accepted = sum(1 for r, _ in rows if (r.get("cosine") or 0.0) >= threshold)
    corpus = html.escape(", ".join(discovery.get("corpus", [])) or discovery.get("mode", ""))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>face-anchor review — {html.escape(probe_label)}</title>
<style>
  :root {{ color-scheme: light dark; --hit: #2e7d32; --bg: Canvas; --fg: CanvasText; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px; background: var(--bg); color: var(--fg); }}
  header {{ display: flex; gap: 20px; align-items: flex-start; margin-bottom: 24px; }}
  header img {{ width: 160px; border-radius: 6px; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  .meta {{ opacity: .75; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; }}
  figure {{ margin: 0; border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius: 6px;
            overflow: hidden; background: color-mix(in srgb, CanvasText 4%, transparent); }}
  figure.hit {{ border-color: var(--hit); border-width: 2px; }}
  figure img {{ display: block; width: 100%; height: 160px; object-fit: cover; }}
  .noimg {{ height: 160px; display: grid; place-items: center; opacity: .5; font-size: 12px; }}
  figcaption {{ display: grid; gap: 2px; padding: 6px 8px; font-size: 12px; }}
  .score {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
  figure.hit .score {{ color: var(--hit); }}
  .verdict, .author {{ opacity: .7; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style>
<header>
  <img src="{probe_thumb}" alt="probe">
  <div>
    <h1>face-anchor review — {html.escape(probe_label)}</h1>
    <div class="meta">
      corpus: {corpus}<br>
      {len(rows)} of {len(fetched)} fetched candidates shown, ordered by cosine<br>
      accept threshold {threshold} — {accepted} candidate(s) at or above it
    </div>
  </div>
</header>
<div class="grid">
{chr(10).join(cells)}
</div>
""")
    return len(rows)
