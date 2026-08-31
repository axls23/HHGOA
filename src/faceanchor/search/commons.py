"""Wikimedia Commons — CC-licensed images, the right corpus for Lane C public figures (PRD §5.2)."""

from __future__ import annotations

import httpx

from faceanchor.search.models import Candidate

BASE_URL = "https://commons.wikimedia.org/w/api.php"
FILE_NAMESPACE = 6

# Commons originals often run into multiple MB (unlike social feed images the
# 512KB fetch cap - PRD §5.2 - was sized for), so a raw Range-capped fetch of
# `url` truncates mid-JPEG and produces visible decode artifacts (observed:
# spurious blur-gate rejections on real portraits during testing). Request a
# server-resized thumbnail instead — same content, comfortably under the cap.
THUMB_WIDTH_PX = 1024


async def search_images(client: httpx.AsyncClient, query: str, limit: int = 20) -> list[Candidate]:
    resp = await client.get(
        BASE_URL,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": limit,
            "gsrnamespace": FILE_NAMESPACE,
            "prop": "imageinfo",
            "iiprop": "url|size|sha1",
            "iiurlwidth": THUMB_WIDTH_PX,
        },
        timeout=5.0,
    )
    resp.raise_for_status()
    data = resp.json()

    pages = (data.get("query") or {}).get("pages", {})
    candidates = []
    for page in pages.values():
        title = page.get("title", "")
        for info in page.get("imageinfo", []):
            image_url = info.get("thumburl") or info.get("url")
            if not image_url:
                continue
            candidates.append(
                Candidate(
                    platform="commons",
                    image_url=image_url,
                    post_uri=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                    text=title,
                    extra={"sha1": info.get("sha1", "")},
                )
            )
    return candidates
