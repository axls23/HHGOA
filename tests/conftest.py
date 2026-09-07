"""Test-wide detector pinning.

Nearly every number in this suite — the 80px floor, the fixture's 78.7px box,
the pose model-point calibration, the exposure ladder — was measured with
YuNet. Letting the suite inherit FACEANCHOR_DETECTOR from the developer's
shell would make those assertions mean different things on different machines,
which is the opposite of what a calibration test is for.

So the default here is pinned, explicitly, and tests that exercise the YOLO
backend opt in with the `yolo_backend` fixture. The pin is about *knowing
which model produced a number*, not a claim that YuNet is the better detector.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def pinned_detector():
    # Session-scoped on purpose: module-scoped fixtures that decode faces run
    # before any function-scoped pin would apply, and would then be built with
    # a different detector than the tests consuming them assert against.
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    patcher.setenv("FACEANCHOR_DETECTOR", "yunet")
    yield
    patcher.undo()


@pytest.fixture
def yolo_backend(monkeypatch):
    from faceanchor.vision import models

    if not models.YOLO_FACE_PATH.exists():
        pytest.skip("YOLOv8-face weights not installed (models/fetch.sh)")
    monkeypatch.setenv("FACEANCHOR_DETECTOR", "yolo")
