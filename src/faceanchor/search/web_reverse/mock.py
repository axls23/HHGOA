"""A replay provider for tests and offline demos — explicitly, never silently.

The one thing this repo must not do is fake a reverse-image search. So this
double is built to be impossible to mistake for the real thing:

- it is only reachable by asking for it by name (`--provider mock`); nothing
  falls back to it, ever, and a missing Google credential raises rather than
  degrading to here;
- it refuses to run without `FACEANCHOR_MOCK_REVERSE_RESPONSE` pointing at a
  response file, so it cannot invent results;
- its `provider_id` is `mock_replay`, and that is what lands in the evidence
  bundle — a replayed run leaves a record that says so, permanently, and its
  digest differs from a real one's;
- `cli.py` prints a warning whenever it is selected.

The file it replays is a real `images:annotate` response body, decoded by the
same `normalize_web_detection` the live provider uses. So the fixture also
exercises the production parser rather than a parallel one written to agree
with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from faceanchor.search.web_reverse.base import (
    MalformedResponseError,
    ProviderConfigError,
    ReverseImageProvider,
    ReverseImageResult,
)
from faceanchor.search.web_reverse.google_vision import normalize_web_detection

RESPONSE_ENV = "FACEANCHOR_MOCK_REVERSE_RESPONSE"
PROVIDER_ID = "mock_replay"


class ReplayProvider(ReverseImageProvider):
    name = "mock"
    provider_id = PROVIDER_ID

    def __init__(self, *, response_path: str | os.PathLike[str] | None = None):
        self._response_path = response_path

    def _resolve_path(self) -> Path:
        raw = self._response_path or os.environ.get(RESPONSE_ENV)
        if not raw:
            raise ProviderConfigError(
                f"--provider mock needs {RESPONSE_ENV} set to a saved images:annotate "
                "response to replay. It has no results of its own and will not invent any."
            )
        path = Path(raw)
        if not path.exists():
            raise ProviderConfigError(f"{RESPONSE_ENV} points at a missing file: {path}")
        return path

    async def search(
        self, image_path: str | os.PathLike[str] | Path, *, client: httpx.AsyncClient | None = None
    ) -> list[ReverseImageResult]:
        # The probe is read but never transmitted — a replay run must still
        # fail on an unreadable probe the way a live one would, and must not
        # send anything anywhere.
        Path(image_path).read_bytes()
        path = self._resolve_path()
        try:
            payload = json.loads(path.read_text())
        except ValueError as e:
            raise MalformedResponseError(f"{path} is not valid JSON: {e}") from e
        return normalize_web_detection(payload, provider=self.provider_id)
