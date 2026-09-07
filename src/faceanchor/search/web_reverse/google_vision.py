"""Google Cloud Vision Web Detection — the real reverse-image arm.

Talks to the official `images:annotate` endpoint with the `WEB_DETECTION`
feature, which is the documented reverse-image surface Google exposes. No
browser automation, no scraping of the Images results page, no text query
built out of the image's metadata — the image bytes themselves go up, base64
inline, and Google's own index answers.

Two credential routes, neither of them ever in the source:

- `GOOGLE_VISION_API_KEY` (or `GOOGLE_CLOUD_VISION_API_KEY`) — an API key,
  sent as the `key` query parameter. Nothing extra to install.
- `GOOGLE_APPLICATION_CREDENTIALS` — a service-account JSON path, exchanged
  for an OAuth2 bearer token by `google-auth` (`pip install -e ".[vision]"`).
  Minting that token means signing a JWT, which is why it is an optional
  dependency rather than a hand-rolled one.

With neither present the provider raises `ProviderAuthError` before reading a
single byte of the probe. That is deliberate and load-bearing: an
unauthenticated run must fail loudly, because the alternative — degrading to
some other kind of search and still calling it reverse-image discovery — is
the specific dishonesty this module exists to avoid.

Response shape (webDetection), and what each bucket means here:

    fullMatchingImages        the same image, elsewhere            -> "full"
    partialMatchingImages     a crop/edit of it                    -> "partial"
    visuallySimilarImages     looks like it; not the same file     -> "similar"
    pagesWithMatchingImages   pages carrying either of the above   -> page_url
    webEntities               text labels for what is in it        -> discarded

`webEntities` is read and dropped on purpose (see base.py): turning those
labels into a text query would be a text search wearing a reverse-image
search's name.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import httpx

from faceanchor.search.web_reverse.base import (
    MalformedResponseError,
    ProviderAuthError,
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ReverseImageProvider,
    ReverseImageResult,
    dedupe_results,
)

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
PROVIDER_ID = "google_web_detection"

API_KEY_ENVS = ("GOOGLE_VISION_API_KEY", "GOOGLE_CLOUD_VISION_API_KEY")
SERVICE_ACCOUNT_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Vision rejects inline `content` over 20MB and degrades well before that;
# 10MB is Google's own documented practical ceiling for annotate requests.
MAX_INLINE_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_RESULTS = 50


def _extract(entries, provider: str, match_type: str) -> list[ReverseImageResult]:
    out = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not url:
            continue
        out.append(ReverseImageResult(provider=provider, image_url=url, page_url=None, match_type=match_type))
    return out


def normalize_web_detection(payload: dict, provider: str = PROVIDER_ID) -> list[ReverseImageResult]:
    """`images:annotate` response -> normalized results. Pure, so it is the
    unit under test and the replay provider's shared decoder.

    Raises `MalformedResponseError` rather than returning `[]` for a body it
    cannot read — an unreadable answer is a provider failure, not a finding
    about the subject. An *empty but well-formed* answer (`{"responses":[{}]}`,
    which is what Vision returns for an image it has never seen) is the one
    case that legitimately yields `[]`.
    """
    if not isinstance(payload, dict):
        raise MalformedResponseError(f"expected a JSON object, got {type(payload).__name__}")

    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise MalformedResponseError("response has no `responses` array")
    first = responses[0]
    if not isinstance(first, dict):
        raise MalformedResponseError("`responses[0]` is not an object")

    # Vision reports per-image failures inside a 200 body rather than as an
    # HTTP status, so this is a provider error that arrives looking like success.
    error = first.get("error")
    if isinstance(error, dict) and (error.get("message") or error.get("code")):
        code = error.get("code", "?")
        if code in (429, 8):  # RESOURCE_EXHAUSTED
            raise ProviderRateLimitError(429, str(error.get("message", "")))
        raise ProviderHTTPError(200, f"annotate error {code}: {error.get('message', '')}")

    web = first.get("webDetection")
    if web is None:
        return []  # well-formed, and the answer is "nothing"
    if not isinstance(web, dict):
        raise MalformedResponseError("`webDetection` is not an object")

    results: list[ReverseImageResult] = []
    results += _extract(web.get("fullMatchingImages"), provider, "full")
    results += _extract(web.get("partialMatchingImages"), provider, "partial")
    results += _extract(web.get("visuallySimilarImages"), provider, "similar")

    for page in web.get("pagesWithMatchingImages") or []:
        if not isinstance(page, dict):
            continue
        page_url = page.get("url")
        if not page_url:
            continue
        title = page.get("pageTitle") or ""
        images = [
            (img.get("url"), kind)
            for kind, key in (("full", "fullMatchingImages"), ("partial", "partialMatchingImages"))
            for img in (page.get(key) or [])
            if isinstance(img, dict)
        ]
        images = [(url, kind) for url, kind in images if url]
        if not images:
            # A page hit with no image URL is still a real hit — the page
            # carries the image, Google just didn't name it. Kept, so the
            # caller can report it as "page without image URL" instead of
            # losing it silently.
            results.append(
                ReverseImageResult(
                    provider=provider, image_url=None, page_url=page_url,
                    match_type="unknown", metadata={"page_title": title},
                )
            )
            continue
        for url, kind in images:
            results.append(
                ReverseImageResult(
                    provider=provider, image_url=url, page_url=page_url,
                    match_type=kind, metadata={"page_title": title},
                )
            )

    return dedupe_results(results)


class GoogleWebDetectionProvider(ReverseImageProvider):
    name = "google"
    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        endpoint: str = ENDPOINT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_results: int = DEFAULT_MAX_RESULTS,
        api_key: str | None = None,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_results = max_results
        self._api_key = api_key  # explicit injection for tests; env is the real route

    # ── credentials ──────────────────────────────────────────────────────
    def _api_key_from_env(self) -> str | None:
        if self._api_key:
            return self._api_key
        for env in API_KEY_ENVS:
            value = os.environ.get(env)
            if value:
                return value
        return None

    @staticmethod
    def _bearer_token_blocking() -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as e:
            raise ProviderAuthError(
                f"{SERVICE_ACCOUNT_ENV} is set but `google-auth` is not installed — "
                'install it with `pip install -e ".[vision]"`, or set '
                f"{API_KEY_ENVS[0]} to use an API key instead"
            ) from e
        try:
            credentials, _ = google.auth.default(scopes=[OAUTH_SCOPE])
            credentials.refresh(Request())
        except Exception as e:  # google-auth raises a wide family here
            raise ProviderAuthError(f"could not mint a token from {SERVICE_ACCOUNT_ENV}: {type(e).__name__}: {e}") from e
        if not credentials.token:
            raise ProviderAuthError("google-auth returned no access token")
        return credentials.token

    async def _auth(self) -> tuple[dict, dict]:
        """(query params, headers). Raises rather than returning unauthenticated."""
        api_key = self._api_key_from_env()
        if api_key:
            return {"key": api_key}, {}
        if os.environ.get(SERVICE_ACCOUNT_ENV):
            token = await asyncio.to_thread(self._bearer_token_blocking)
            return {}, {"Authorization": f"Bearer {token}"}
        raise ProviderAuthError(
            "no Google Cloud Vision credentials found. Set one of: "
            f"{API_KEY_ENVS[0]} (an API key), or {SERVICE_ACCOUNT_ENV} (a service-account "
            'JSON path, with `pip install -e ".[vision]"`). Reverse-image search does not '
            "run without them — there is no offline fallback, by design."
        )

    # ── search ───────────────────────────────────────────────────────────
    async def search(
        self, image_path: str | os.PathLike[str] | Path, *, client: httpx.AsyncClient | None = None
    ) -> list[ReverseImageResult]:
        path = Path(image_path)
        raw = path.read_bytes()
        if not raw:
            raise MalformedResponseError(f"{path} is empty — nothing to search with")
        if len(raw) > MAX_INLINE_BYTES:
            raise ProviderHTTPError(
                413, f"probe is {len(raw)} bytes, over the {MAX_INLINE_BYTES}-byte inline limit for images:annotate"
            )

        params, headers = await self._auth()
        body = {
            "requests": [
                {
                    "image": {"content": base64.b64encode(raw).decode("ascii")},
                    "features": [{"type": "WEB_DETECTION", "maxResults": self.max_results}],
                }
            ]
        }

        owned = client is None
        http = client or httpx.AsyncClient(http2=True)
        try:
            resp = await http.post(
                self.endpoint, params=params, headers=headers, json=body, timeout=self.timeout_s
            )
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"provider did not answer within {self.timeout_s}s: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderHTTPError(0, f"{type(e).__name__}: {e}") from e
        finally:
            if owned:
                await http.aclose()

        if resp.status_code in (401, 403):
            raise ProviderAuthError(f"provider rejected the credentials (HTTP {resp.status_code}): {_snippet(resp)}")
        if resp.status_code == 429:
            raise ProviderRateLimitError(429, _snippet(resp))
        if resp.status_code >= 400:
            raise ProviderHTTPError(resp.status_code, _snippet(resp))

        try:
            payload = resp.json()
        except ValueError as e:
            raise MalformedResponseError(f"provider returned non-JSON: {e}") from e

        return normalize_web_detection(payload, provider=self.provider_id)


def _snippet(resp: httpx.Response, limit: int = 200) -> str:
    """A short, non-sensitive slice of an error body.

    Never the request: the request carries the probe image. Only the
    provider's own complaint comes back into a log line.
    """
    try:
        return resp.text[:limit].replace("\n", " ")
    except Exception:
        return "<unreadable body>"
