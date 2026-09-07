"""The documents are the spec; a path in them that does not exist is a lie.

`README.md` claimed `contracts/src/chain/solana.py` for most of this build —
a file that has never existed at that path. Nothing catches that class of
error, because a wrong path in prose still reads perfectly. This does.

Only paths the repo is supposed to *contain* are checked. Build output
(`contracts/out/`, `zk/build/`), fetched weights (`models/*.onnx`), runtime
artifacts (`evidence/`) and gitignored secrets (`consent/*.json`) are named
in the docs on purpose and are absent on purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/ARCHITECTURE.md", "docs/THREAT_MODEL.md", "docs/CONSENT.md"]

_PATH = re.compile(
    r"\b(?:src/faceanchor|contracts|tests|scripts|zk|models)/[A-Za-z0-9_./-]+"
    r"\.(?:py|sol|circom|sh|json|jsonl|toml)\b"
)

# ARCHITECTURE.md talks in packages as much as files ("src/faceanchor/vision/"),
# and a renamed package is the same class of error as a renamed file.
_DIR = re.compile(r"\bsrc/faceanchor/[A-Za-z0-9_/]+/")

# Named in the docs, absent by design: generated, fetched, or ignored.
_GENERATED = ("contracts/out/", "contracts/broadcast/", "zk/build/", "models/", "node_modules/")


def _referenced_paths(text: str) -> set[str]:
    found = set(_PATH.findall(text)) | set(_DIR.findall(text))
    return {m for m in found if not m.startswith(_GENERATED)}


def test_documented_paths_exist():
    missing: list[str] = []
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text()
        for referenced in sorted(_referenced_paths(text)):
            if not (REPO_ROOT / referenced).exists():
                missing.append(f"{doc}: {referenced}")
    assert not missing, "documented paths that do not exist:\n  " + "\n  ".join(missing)


def test_the_guard_would_catch_a_wrong_path():
    """The regression this exists for, asserted against the actual old text."""
    assert _referenced_paths("`contracts/src/chain/solana.py` (SPL Memo)") == {"contracts/src/chain/solana.py"}
    assert not (REPO_ROOT / "contracts/src/chain/solana.py").exists()
