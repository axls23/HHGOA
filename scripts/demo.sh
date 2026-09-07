#!/usr/bin/env bash
# End-to-end judge-facing demo: every claim in README's "Demo" section, run
# in order against a local Anvil chain, with the expected outcome asserted
# for each act (including the two acts that are supposed to FAIL).
#
#   ./scripts/demo.sh --query "official portrait headshot high resolution"
#
# With no --probe it takes the probe from your webcam, with Stage 1's quality
# gate running live on the preview so the frame you keep is one the pipeline
# has already agreed to accept.
#
# Nothing here is mocked: the discovery stage hits live public APIs, the
# anchor is a real transaction, and the tamper/forge acts fail for real
# reasons. See README "Demo" and PRD §12.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

export PATH="$HOME/.foundry/bin:$PATH"
VENV="$REPO_ROOT/.venv/bin"
FA="$VENV/python -m faceanchor.cli"
RPC_URL="http://127.0.0.1:8545"
# Foundry dev key #0 — anvil-only, funded automatically, never a real wallet.
ANVIL_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

PROBE=""
# Take the probe out of a corpus database instead of the webcam. Only images
# you imported yourself (`corpus-build --from-dir`, provenance "local") are
# eligible — see extract_probe_from_corpus() for why a scraped stranger is not.
PROBE_FROM_CORPUS=""
PROBE_PICK=""
PROBE_FROM_CORPUS_OUT="evidence/probe-from-corpus.jpg"
CAMERA=0
CAPTURE_ARGS=()
CAPTURED=0
NO_LIVENESS=0
CONSENT="consent/self.example.json"
CONSENT_GIVEN=0
# Lane A (docs/CONSENT.md): the operator's own face. Shown in full before the
# camera opens, because consent to a statement nobody read is not consent.
CONSENT_STATEMENT="I consent to my own face being used as the probe image for this face-anchor pipeline run, including the discovery search and on-chain anchoring described in PRD_face_anchor.md."
QUERY="official portrait headshot high resolution"
QUERY_GIVEN=0
# Discovery arm: byface | text | web-reverse. These are three different
# questions (see README "Three discovery modes"); exactly one runs per demo.
MODE=byface
PROVIDER=google
# A locally built corpus database replaces the live fetch entirely. Point the
# demo at one when the network is restricted, when a repeatable run matters
# more than a fresh one, or when re-downloading strangers' images on every
# rehearsal is the wrong thing to do.
CORPUS_DB=""
CORPORA=(mastodon:tag/selfie mastodon:tag/portrait@mstdn.social bsky:feed/whats-hot)
CORPUS_GIVEN=0
HANDLE=""
SEED_QUERY=""
CONTRACT=""
DO_ZK=1
PAUSE=1
KEEP_ANVIL=0
ANVIL_PID=""
ANVIL_LOG="$REPO_ROOT/evidence/anvil-demo.log"
CANDIDATE_LOG="evidence/candidates.jsonl"
CONTACT_SHEET="evidence/review.html"

usage() {
    cat <<'USAGE'
usage: scripts/demo.sh [--probe PHOTO] [options]

  --probe PATH        use an existing photo instead of the webcam
  --probe-from-corpus DIR
                      take the probe out of a corpus database instead of the
                      webcam — repeatable, no camera, same face every run.
                      Only images you imported yourself are eligible
                      (corpus-build --from-dir, provenance "local"): making a
                      scraped stranger the subject of a live face search is
                      the one thing this tool must not do (docs/CONSENT.md)
  --probe-pick SEL    which image: a sha256 prefix, or a 0-based index
                      [the first eligible one]
  --live              search live public timelines rather than replaying a
                      corpus database (the default; use this to override an
                      earlier --corpus-db)
  --camera N          webcam device index    [0]
  --headless-capture  no preview window: auto-take the first gate-passing frame
  --no-liveness       development: capture without the anti-spoofing check, so
                      a photo or a face on a screen is accepted as the probe.
                      Nothing downstream can tell — the bundle records the
                      probe's hash, not how it was captured
  --consent PATH      consent artifact. Without this, a webcam capture asks
                      you to sign one interactively before the camera opens
                      and writes it to consent/self.json (gitignored)
  --handle H          your own account, added to the corpus so the search can
                      find you: @user@instance | user.bsky.social
  --corpus SPEC       extra corpus to search, repeatable
                      [mastodon:tag/selfie, mastodon:tag/portrait@mstdn.social,
                       bsky:feed/whats-hot]
  --corpus-db DIR     run the pipeline over a locally built corpus database
                      instead of fetching live timelines. Offline, repeatable,
                      and nothing explicit is re-downloaded — the safety filter
                      already ran at build time. Build one with:
                        faceanchor corpus-build --db DIR --corpus mastodon:tag/selfie
  --query "..."       opt out of face-embedding corpus search and use
                      text-seeded discovery instead
                      ["official portrait headshot high resolution"]
  --web-reverse       genuine image-based reverse-image search: the probe
                      image is submitted to an external provider, and the
                      social-media results it returns are verified by face
  --provider NAME     reverse-image provider   [google]
                      google -> Cloud Vision Web Detection, needs
                        GOOGLE_VISION_API_KEY or GOOGLE_APPLICATION_CREDENTIALS
                      mock   -> replays a saved response file named by
                        FACEANCHOR_MOCK_REVERSE_RESPONSE (offline; the bundle
                        records the provider as mock_replay, permanently)
  --seed-query "..."  explicit Arm B seed    [derived from Arm A's best hit]
  --contract ADDR     reuse a deployed EvidenceAnchor instead of deploying
  --rpc URL           RPC endpoint           [http://127.0.0.1:8545]
  --no-zk             skip acts 8-9 (Groth16 proof + forged-score rejection)
  --no-pause          don't wait for Enter between acts (CI / recording)
  --keep-anvil        leave a demo-started anvil running at exit
  -h, --help          this message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe)       PROBE="$2"; shift 2 ;;
        --probe-from-corpus) PROBE_FROM_CORPUS="$2"; shift 2 ;;
        --probe-pick)  PROBE_PICK="$2"; shift 2 ;;
        --live)        CORPUS_DB=""; MODE=byface; shift ;;
        --camera)      CAMERA="$2"; shift 2 ;;
        --headless-capture) CAPTURE_ARGS+=(--no-preview); shift ;;
        --no-liveness) CAPTURE_ARGS+=(--no-liveness); NO_LIVENESS=1; shift ;;
        --consent)     CONSENT="$2"; CONSENT_GIVEN=1; shift 2 ;;
        --query)       QUERY="$2"; QUERY_GIVEN=1; MODE=text; shift 2 ;;
        --by-face)     MODE=byface; shift ;;
        --corpus-db)   CORPUS_DB="$2"; MODE=byface; shift 2 ;;
        --web-reverse) MODE=web-reverse; shift ;;
        --provider)    PROVIDER="$2"; MODE=web-reverse; shift 2 ;;
        --handle)      HANDLE="$2"; shift 2 ;;
        --corpus)      if (( ! CORPUS_GIVEN )); then CORPORA=(); CORPUS_GIVEN=1; fi
                       CORPORA+=("$2"); MODE=byface; shift 2 ;;
        --seed-query)  SEED_QUERY="$2"; shift 2 ;;
        --contract)    CONTRACT="$2"; shift 2 ;;
        --rpc)         RPC_URL="$2"; shift 2 ;;
        --no-zk)       DO_ZK=0; shift ;;
        --no-pause)    PAUSE=0; shift ;;
        --keep-anvil)  KEEP_ANVIL=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

BY_FACE=0; [[ "$MODE" == byface ]] && BY_FACE=1

if [[ -t 1 ]]; then
    B=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYA=$'\033[36m'; R=$'\033[0m'
else
    B=""; DIM=""; RED=""; GRN=""; YEL=""; CYA=""; R=""
fi

ACT=0
FAILURES=()
SKIPPED=()

act()  { ACT=$((ACT + 1)); printf '\n%s\n%s── Act %d · %s%s\n' "$DIM$(printf '─%.0s' {1..72})$R" "$B$CYA" "$ACT" "$1" "$R"; [[ -n "${2:-}" ]] && printf '%s%s%s\n' "$DIM" "$2" "$R"; }
note() { printf '%s%s%s\n' "$DIM" "$1" "$R"; }
ok()   { printf '%s  ✓ %s%s\n' "$GRN" "$1" "$R"; }
bad()  { printf '%s  ✗ %s%s\n' "$RED" "$1" "$R"; FAILURES+=("$1"); }
skip() { printf '%s  ⊘ %s%s\n' "$YEL" "$1" "$R"; SKIPPED+=("$1"); }
die()  { printf '\n%s✗ %s%s\n' "$RED" "$1" "$R" >&2; exit 1; }

# Echo a command the way a human would type it, then run it.
show() { printf '\n%s$ %s%s\n' "$B" "$*" "$R"; }
runv() { show "$@"; "$@"; }

pause() {
    (( PAUSE )) || return 0
    [[ -t 0 ]] || return 0
    printf '\n%s[Enter] to continue%s' "$DIM" "$R"
    read -r _
}

# Development view of the search: every URL looked at, and why it lost.
_show_candidate_log() {
    if [[ -f "$CONTACT_SHEET" ]]; then
        note "the images themselves, probe first, ordered by score — open $CONTACT_SHEET in a browser"
        note "  a score above the threshold is not a verified match until a human has looked at it"
    fi
    [[ -f "$CANDIDATE_LOG" ]] || return 0
    note "every URL this run searched, against the probe — $CANDIDATE_LOG"
    show "head -1 $CANDIDATE_LOG | python -m json.tool   # probe + corpora + per-source counts"
    "$VENV/python" -c "import json,sys;print(json.dumps(json.loads(open('$CANDIDATE_LOG').readline()),indent=2))"
    show "verdict breakdown"
    "$VENV/python" - "$CANDIDATE_LOG" <<'PYLOG'
import collections, json, sys
lines = open(sys.argv[1]).read().splitlines()[1:]
records = [json.loads(x) for x in lines]
counts = collections.Counter(r.get("verdict", "?") for r in records)
for verdict, n in counts.most_common():
    print(f"  {n:5d}  {verdict}")
scored = [r for r in records if r.get("cosine") is not None]
if scored:
    top = sorted(scored, key=lambda r: r["cosine"], reverse=True)[:5]
    print("  top cosine scores (threshold 0.363):")
    for r in top:
        print(f"    {r['cosine']:.4f}  {r['platform']:9s} {r['image_url'][:64]}")
PYLOG
}

# Pull a probe image out of a corpus database.
#
# The eligibility rule is the whole point of this function. A corpus built
# from public timelines is a few hundred strangers who posted a photo, not
# subjects who agreed to be searched for — and making one of them the probe
# turns this demo into exactly the untargeted face-identification the project
# refuses to be (PRD §3, docs/CONSENT.md, EU AI Act Art. 5(1)(e)). Images you
# imported yourself carry provenance "local", and those are the ones a consent
# artifact can honestly cover, so those are the ones this will hand you.
extract_probe_from_corpus() {
    "$VENV/python" - "$PROBE_FROM_CORPUS" "$PROBE_PICK" "$PROBE_FROM_CORPUS_OUT" <<'PYPICK'
import json
import shutil
import sys
from pathlib import Path

db, selector, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
manifest = db / "manifest.jsonl"
if not manifest.exists():
    print(f"no corpus database at {db} (expected manifest.jsonl)", file=sys.stderr)
    raise SystemExit(2)

rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
entries = [r for r in rows[1:] if r.get("sha256")]
eligible = [e for e in entries if e.get("platform") == "local"]

REFUSAL = 3


def refuse(entry):
    print(
        f"that image came from {entry['platform']} — {entry.get('post_uri', '?')}\n"
        "It is a stranger who posted a photo, not a subject who agreed to be searched for.\n"
        "Using it as the probe would make this an untargeted face search against a real\n"
        "person, which is the one thing this pipeline refuses to be (docs/CONSENT.md).\n"
        "\n"
        "Use an image you are entitled to search for:\n"
        "  faceanchor corpus-build --db my-corpus --from-dir /path/to/your/photos\n"
        "  scripts/demo.sh --probe-from-corpus my-corpus",
        file=sys.stderr,
    )
    raise SystemExit(REFUSAL)


if selector:
    # Resolve against every entry, not just the eligible ones, so that asking
    # for a scraped image by name gets the refusal and its reason rather than
    # a misleading "not found".
    if selector.isdigit() and int(selector) < len(eligible):
        chosen = eligible[int(selector)]
    else:
        matches = [e for e in entries if e["sha256"].startswith(selector)]
        if not matches:
            print(f"no image in {db} matches {selector!r}", file=sys.stderr)
            raise SystemExit(2)
        chosen = matches[0]
        if chosen.get("platform") != "local":
            refuse(chosen)
elif eligible:
    chosen = eligible[0]
else:
    print(
        f"{db} holds {len(entries)} image(s), none of them yours: every one was fetched from a\n"
        "public timeline. Import photos you are entitled to use first:\n"
        "  faceanchor corpus-build --db my-corpus --from-dir /path/to/your/photos",
        file=sys.stderr,
    )
    raise SystemExit(REFUSAL)

src = db / "images" / chosen["sha256"][:2] / chosen["sha256"]
if not src.exists():
    print(f"manifest names {chosen['sha256']} but {src} is missing", file=sys.stderr)
    raise SystemExit(2)
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, out)
print(f"{out}\t{chosen['sha256'][:16]}\t{chosen.get('width')}x{chosen.get('height')}\t{chosen.get('post_uri', '')}")
print(f"{len(eligible)} eligible of {len(entries)} in the corpus", file=sys.stderr)
PYPICK
}

# Collect Lane A consent before the shutter, not after.
#
# Only for the webcam path: that is the one case where the subject is certainly
# the person at the keyboard, so they are the one who can consent. A --probe or
# --probe-from-corpus run keeps requiring an artifact you supply, because this
# script has no idea whose face is in a file.
collect_self_consent() {
    local target="consent/self.json" who confirm existing_who existing_when

    if [[ ! -t 0 ]]; then
        note "stdin is not a terminal — keeping --consent $CONSENT, no prompt"
        return 0
    fi

    if [[ -f "$target" ]]; then
        existing_who=$("$VENV/python" -c "import json;print(json.load(open('$target')).get('signed_by',''))" 2>/dev/null)
        existing_when=$("$VENV/python" -c "import json;print(json.load(open('$target')).get('signed_at',''))" 2>/dev/null)
        printf '\n%sfound %s — signed by %s at %s%s\n' "$DIM" "$target" "${existing_who:-?}" "${existing_when:-?}" "$R"
        printf '%sreuse it?%s [Y/n] ' "$B" "$R"
        read -r reuse
        if [[ ! "$reuse" =~ ^[Nn] ]]; then
            CONSENT="$target"
            ok "consent: $CONSENT (reused)"
            return 0
        fi
    fi

    printf '\n%s%s%s\n' "$B" "$CONSENT_STATEMENT" "$R"
    note "Lane A, docs/CONSENT.md. Only this artifact's SHA-256 digest is anchored — the file"
    note "itself never leaves this machine, and consent/self.json is gitignored."
    note "You can withdraw it later: faceanchor revoke --consent $target"

    printf '\n%syour name or email%s (Enter to abort): ' "$B" "$R"
    read -r who
    [[ -n "$who" ]] || die "no consent given — nothing was captured, searched or anchored"

    # The phrase goes on its own line, in quotes. Running it into the sentence
    # ("type I consent to sign") reads as an instruction to type the whole
    # thing, which is a prompt bug, not a user error. Case and surrounding
    # whitespace are ignored, and a mismatch costs a retry rather than the run:
    # deliberate is the point, punishing a typo is not.
    local attempt normalized
    for attempt in 1 2; do
        printf '\n%sTo sign, type exactly:%s  %sI consent%s\n' "$B" "$R" "$CYA" "$R"
        printf '%s(Enter alone aborts)%s\n> ' "$DIM" "$R"
        read -r confirm
        normalized=$(printf '%s' "$confirm" | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//')
        [[ "$normalized" == "i consent" ]] && break
        [[ -z "$normalized" ]] && die "not signed — nothing was captured, searched or anchored"
        if (( attempt == 2 )); then
            die "not signed — nothing was captured, searched or anchored"
        fi
        printf '%s  that was "%s" — the phrase is just the two words: I consent%s\n' "$YEL" "$confirm" "$R"
    done

    "$VENV/python" - "$target" "$who" "$CONSENT_STATEMENT" <<'PYCONSENT'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

target, signed_by, statement = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not statement.strip() or not signed_by.strip():
    # A consent artifact with no statement is a signature on nothing. Better to
    # write no file at all than one that looks valid and says nothing.
    print("refusing to write a consent artifact with an empty statement or signatory", file=sys.stderr)
    raise SystemExit(2)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(
        {
            "lane": "A",
            "subject": "self",
            "statement": statement,
            "signed_by": signed_by,
            "signed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        indent=2,
    )
    + "\n"
)
PYCONSENT
    [[ -f "$target" ]] || die "could not write $target"
    CONSENT="$target"
    ok "consent signed by $who -> $CONSENT"
    show "sha256 of the canonicalised artifact — this is what gets anchored"
    "$VENV/python" -c "from faceanchor.consent import load_consent_digest;print(load_consent_digest('$target'))"
}

cleanup() {
    if [[ -n "$ANVIL_PID" ]] && (( ! KEEP_ANVIL )); then
        kill "$ANVIL_PID" 2>/dev/null
        note "stopped the anvil this demo started (pid $ANVIL_PID)"
    fi
}
trap cleanup EXIT

rpc_up() {
    curl -s -m 2 -X POST -H 'Content-Type: application/json' \
        --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
        "$RPC_URL" 2>/dev/null | grep -q result
}

# ── Act 1 · preflight ────────────────────────────────────────────────────
act "Preflight" "everything below is checked before a single stage runs"

if [[ -n "$PROBE" && -n "$PROBE_FROM_CORPUS" ]]; then
    die "--probe and --probe-from-corpus both name a probe — pick one"
fi
if [[ -n "$PROBE" ]]; then
    [[ -f "$PROBE" ]] || die "probe image not found: $PROBE"
elif [[ -n "$PROBE_FROM_CORPUS" ]]; then
    [[ -f "$PROBE_FROM_CORPUS/manifest.jsonl" ]] \
        || die "no corpus database at $PROBE_FROM_CORPUS (expected manifest.jsonl). Build one: faceanchor corpus-build --db $PROBE_FROM_CORPUS --from-dir /path/to/your/photos"
else
    compgen -G '/dev/video*' >/dev/null \
        || die "no webcam found at /dev/video* — pass --probe PHOTO, or --probe-from-corpus DIR, to use an existing image instead"
fi
[[ -f "$CONSENT" ]] || die "consent artifact not found: $CONSENT"
[[ -x "$VENV/python" ]] || die "no venv at $VENV — run 'make setup' first"
command -v forge >/dev/null || die "forge not found — run 'make contracts' (installs Foundry)"
command -v anvil >/dev/null || die "anvil not found — run 'foundryup'"

MODEL_CACHE="${FACEANCHOR_MODEL_CACHE:-$HOME/.cache/faceanchor/models}"
[[ -f "$MODEL_CACHE/face_detection_yunet_2023mar.onnx" && -f "$MODEL_CACHE/face_recognition_sface_2021dec.onnx" ]] \
    || die "ONNX weights missing from $MODEL_CACHE — run './models/fetch.sh'"
if   [[ -n "$PROBE" ]];             then PROBE_DESC="$PROBE"
elif [[ -n "$PROBE_FROM_CORPUS" ]]; then PROBE_DESC="<corpus $PROBE_FROM_CORPUS>"
else                                     PROBE_DESC="<webcam $CAMERA>"
fi
ok "probe=$PROBE_DESC  consent=$CONSENT  models=$MODEL_CACHE"

# Reverse-image search finds you only if you are in the corpus. The default
# corpora are public hashtag timelines and feeds — a few hundred strangers.
# Ask for the one corpus that is guaranteed to contain you, before the camera
# opens and a live search goes by.
if [[ -n "$CORPUS_DB" ]]; then
    [[ -f "$CORPUS_DB/manifest.jsonl" ]] \
        || die "no corpus database at $CORPUS_DB (expected manifest.jsonl). Build one first: faceanchor corpus-build --db $CORPUS_DB --corpus mastodon:tag/selfie"
    DB_IMAGES=$("$VENV/python" -c "import json,sys;print(json.loads(open('$CORPUS_DB/manifest.jsonl').readline())['images'])" 2>/dev/null || echo "?")
    ok "corpus database: $CORPUS_DB ($DB_IMAGES images) — this run is OFFLINE, no live fetch"
    note "the images were filtered for explicit content at build time and are replayed byte-for-byte"
fi

if (( BY_FACE )) && [[ -z "$CORPUS_DB" ]] && [[ -z "$HANDLE" ]]; then
    printf '\n%sFace-embedding corpus search only finds you if you are in the corpus.%s\n' "$YEL" "$R"
    note "Default corpora are public timelines/feeds — ~400 strangers, none of them you."
    note "Give a handle you have posted a photo of your face to, and it is added as a corpus:"
    note "  Mastodon @you@instance   ·   Bluesky you.bsky.social"
    if [[ -t 0 ]]; then
        printf '\n%shandle%s (Enter to skip — the search then honestly finds nothing): ' "$B" "$R"
        read -r TYPED
        [[ -n "$TYPED" ]] && HANDLE="$TYPED"
    fi
fi

if [[ -n "$HANDLE" ]]; then
    # Shape decides the platform: "@user@instance"/"@user" is Mastodon,
    # "name.tld" is a Bluesky handle.
    case "$HANDLE" in
        @*)  CORPORA+=("mastodon:account/${HANDLE#@}") ;;
        *.*) CORPORA+=("bsky:author/$HANDLE") ;;
        *)   CORPORA+=("mastodon:account/$HANDLE") ;;
    esac
    ok "corpus includes you: ${CORPORA[-1]}"
fi

if [[ "$MODE" == web-reverse ]]; then
    case "$PROVIDER" in
        google)
            if [[ -z "${GOOGLE_VISION_API_KEY:-}${GOOGLE_CLOUD_VISION_API_KEY:-}${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
                die "--web-reverse --provider google needs real credentials: export GOOGLE_VISION_API_KEY=<key>, or GOOGLE_APPLICATION_CREDENTIALS=<service-account.json> (plus 'pip install -e \".[vision]\"'). There is no offline fallback — a fake reverse-image search would be worse than none. For an offline dry run use --provider mock with FACEANCHOR_MOCK_REVERSE_RESPONSE."
            fi
            ok "reverse-image provider: Google Cloud Vision Web Detection (credentials present)"
            ;;
        mock)
            [[ -n "${FACEANCHOR_MOCK_REVERSE_RESPONSE:-}" ]] \
                || die "--provider mock needs FACEANCHOR_MOCK_REVERSE_RESPONSE pointing at a saved images:annotate response to replay"
            [[ -f "$FACEANCHOR_MOCK_REVERSE_RESPONSE" ]] \
                || die "FACEANCHOR_MOCK_REVERSE_RESPONSE points at a missing file: $FACEANCHOR_MOCK_REVERSE_RESPONSE"
            printf '\n%sthis run REPLAYS a saved provider response — it is not a live reverse-image search.%s\n' "$YEL" "$R"
            note "the evidence bundle will record provider=mock_replay, so the record says so permanently"
            ;;
        *) die "unknown --provider $PROVIDER (google | mock)" ;;
    esac
fi

if (( DO_ZK )) && [[ ! -f "$REPO_ROOT/zk/build/match_final.zkey" ]]; then
    skip "acts 8-9 (ZK): zk/build/match_final.zkey missing — run 'cd zk && ./setup.sh'"
    DO_ZK=0
fi

if rpc_up; then
    ok "chain reachable at $RPC_URL"
else
    [[ "$RPC_URL" == "http://127.0.0.1:8545" ]] || die "no RPC at $RPC_URL"
    note "no chain at $RPC_URL — starting anvil..."
    mkdir -p "$(dirname "$ANVIL_LOG")"
    # Not --silent: the log is already redirected to a file, so it costs the
    # terminal nothing, and "anvil failed to come up — see $ANVIL_LOG" is
    # useless if --silent guaranteed that file is empty.
    anvil >"$ANVIL_LOG" 2>&1 &
    ANVIL_PID=$!
    for _ in $(seq 1 40); do rpc_up && break; sleep 0.25; done
    rpc_up || die "anvil failed to come up — see $ANVIL_LOG"
    ok "anvil started (pid $ANVIL_PID, log $ANVIL_LOG)"
fi
pause

# ── Act 2 · the probe ────────────────────────────────────────────────────
if [[ -n "$PROBE_FROM_CORPUS" ]]; then
    act "The probe" "taken from a corpus database — no camera, and the same face every run"
    show "probe <- $PROBE_FROM_CORPUS ${PROBE_PICK:+(pick $PROBE_PICK)}"
    PICKED=$(extract_probe_from_corpus)
    PICK_RC=$?
    if (( PICK_RC == 3 )); then
        die "refused to use that image as the probe (see above)"
    elif (( PICK_RC != 0 )); then
        die "could not take a probe out of $PROBE_FROM_CORPUS"
    fi
    PROBE=$(cut -f1 <<<"$PICKED")
    ok "probe: $PROBE  sha256=$(cut -f2 <<<"$PICKED")…  $(cut -f3 <<<"$PICKED")  from $(cut -f4 <<<"$PICKED")"
    note "eligible because you imported it yourself (provenance: local). $CONSENT must name"
    note "  this subject — the artifact is what makes the run lawful, not the file path."
elif [[ -n "$PROBE" ]]; then
    act "The probe" "using the photo you passed in"
    ok "probe: $PROBE"
else
    act "Consent, then the probe" "PRD §3: the artifact is signed before the camera opens"
    (( CONSENT_GIVEN )) && note "using the artifact you passed: $CONSENT" || collect_self_consent
    pause
    act "The probe" "taken live from the webcam, with Stage 1's quality gate on the preview"
    PROBE="evidence/probe.jpg"
    (( NO_LIVENESS )) && note "anti-spoofing is OFF for this capture (--no-liveness): a photo or a screen will be accepted"
    note "SPACE captures (only once the gate goes green) · ESC aborts · F forces a failing frame"
    note "the shutter is the consent moment, and $CONSENT is already signed for it"
    show "python scripts/capture_probe.py --out $PROBE --camera $CAMERA ${CAPTURE_ARGS[*]}"
    "$VENV/python" scripts/capture_probe.py --out "$PROBE" --camera "$CAMERA" "${CAPTURE_ARGS[@]+"${CAPTURE_ARGS[@]}"}"
    CAP_RC=$?
    (( CAP_RC == 2 )) && die "capture aborted — nothing was written, nothing was searched, nothing was anchored"
    (( CAP_RC == 0 )) || die "webcam capture failed (try --camera 1, or --probe PHOTO to use an existing image)"
    CAPTURED=1
    ok "probe captured and already through the gate the pipeline will re-apply"
    note "$PROBE is gitignored (evidence/*) — it is a face image, keep it that way"
fi
pause

# ── Act 3 · the consent gate ─────────────────────────────────────────────
act "The consent gate" "PRD §3: a run without an explicit consent artifact does not happen"
show "faceanchor run --probe $PROBE --query '...'   # note: no --subject-consent"
GATE_OUT=$($FA run --probe "$PROBE" --query "$QUERY" --chain anvil --contract-address 0x0000000000000000000000000000000000000000 2>&1)
GATE_RC=$?
echo "$GATE_OUT" | head -3
if (( GATE_RC != 0 )) && grep -q 'subject-consent is required' <<<"$GATE_OUT"; then
    ok "refused before touching the image, the network, or the chain"
else
    bad "the pipeline ran without consent — the §3 gate is not enforcing"
fi
pause

# ── Act 4 · deploy ───────────────────────────────────────────────────────
act "Deploy" "EvidenceAnchor stores only 32-byte digests — no personal data ever reaches it"
if [[ -n "$CONTRACT" ]]; then
    ok "reusing EvidenceAnchor at $CONTRACT"
else
    show "forge script script/Deploy.s.sol --rpc-url $RPC_URL --private-key \$ANVIL_KEY --broadcast"
    DEPLOY_OUT=$(cd contracts && forge script script/Deploy.s.sol --rpc-url "$RPC_URL" --private-key "$ANVIL_KEY" --broadcast 2>&1)
    echo "$DEPLOY_OUT" | grep -E 'deployed at:' || echo "$DEPLOY_OUT" | tail -20
    CONTRACT=$(echo "$DEPLOY_OUT" | grep 'EvidenceAnchor deployed at:' | tail -1 | grep -oE '0x[0-9a-fA-F]{40}')
    [[ -n "$CONTRACT" ]] || die "could not parse the EvidenceAnchor address out of forge's output"
    ok "EvidenceAnchor at $CONTRACT"
fi
CHAIN_ARGS=(--chain anvil --contract-address "$CONTRACT")
pause

# ── Act 5 · the pipeline ─────────────────────────────────────────────────
if (( BY_FACE )) && [[ -z "$CORPUS_DB" ]]; then
    act "Stage 1 → 2 → 3" "detect+embed the face, search the live decentralized networks by face, anchor the finding"
else
    act "Stage 1 → 2 → 3" "detect+embed the face, search, anchor the finding"
fi
RUN_ARGS=(run --probe "$PROBE" --subject-consent "$CONSENT" --log-candidates "$CANDIDATE_LOG" --contact-sheet "$CONTACT_SHEET" "${CHAIN_ARGS[@]}")
if [[ "$MODE" == web-reverse ]]; then
    RUN_ARGS+=(--discovery web-reverse --provider "$PROVIDER" --accept-external-upload)
    note "reverse-image mode: the PROBE IMAGE ITSELF is submitted to $PROVIDER — a trust boundary the"
    note "other two modes never cross. Results are then filtered to social platforms and each surviving"
    note "candidate is verified by face; a provider hit alone is never treated as a match."
    show "faceanchor run --probe $PROBE --subject-consent $CONSENT --discovery web-reverse --provider $PROVIDER --accept-external-upload ${CHAIN_ARGS[*]}"
elif (( BY_FACE )) && [[ -n "$CORPUS_DB" ]]; then
    RUN_ARGS+=(--by-face --corpus-db "$CORPUS_DB")
    note "offline mode: the corpus was compiled once and is replayed byte-for-byte — no network,"
    note "  no re-download of strangers' images, and the same input every run. The bundle pins the"
    note "  corpus by its manifest sha256, so it records exactly what was searched."
    show "faceanchor run --probe $PROBE --subject-consent $CONSENT --by-face --corpus-db $CORPUS_DB --log-candidates $CANDIDATE_LOG ${CHAIN_ARGS[*]}"
elif (( BY_FACE )); then
    RUN_ARGS+=(--by-face)
    for spec in "${CORPORA[@]}"; do RUN_ARGS+=(--corpus "$spec"); done
    note "LIVE face search across decentralized networks: Mastodon instances (ActivityPub) and"
    note "  Bluesky (AT Protocol) are queried right now — ${#CORPORA[@]} corpora, fetched this second."
    note "face-embedding mode: no text about the subject is sent anywhere — the probe's own face is the query"
    note "public timelines carry explicit content: posts flagged sensitive (Mastodon CW) or labelled"
    note "  explicit (Bluesky) are dropped before they are fetched, and the count is printed below"
    show "faceanchor run --probe $PROBE --subject-consent $CONSENT --by-face ${CORPORA[*]/#/--corpus } --log-candidates $CANDIDATE_LOG ${CHAIN_ARGS[*]}"
else
    RUN_ARGS+=(--query "$QUERY")
    [[ -n "$SEED_QUERY" ]] && RUN_ARGS+=(--seed-query "$SEED_QUERY")
    show "faceanchor run --probe $PROBE --subject-consent $CONSENT --query '$QUERY' ${CHAIN_ARGS[*]}"
fi
RUN_LOG=$(mktemp)
$FA "${RUN_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
RUN_RC=${PIPESTATUS[0]}
RUN_OUT=$(cat "$RUN_LOG"); rm -f "$RUN_LOG"
if (( RUN_RC != 0 )); then
    if (( RUN_RC == 2 )); then
        # Exit 2 is PROVIDER_ERROR, and it is deliberately not exit 1. The
        # search did not happen, so this run learned nothing about the
        # subject — reporting it as "no match" would be the worst lie this
        # tool could tell.
        die "PROVIDER_ERROR — the reverse-image provider did not answer (see above). This is NOT 'the face was not found': the search never ran. Check credentials, quota and connectivity, then re-run."
    elif grep -q 'quality gate failed' <<<"$RUN_OUT"; then
        die "Stage 1 rejected the probe before any search ran — that gate is doing its job (PRD §5.3), see the reason above. It wants a face at least 80px on its short side, in focus, roughly head-on."
    elif grep -qE 'multiple faces|no face' <<<"$RUN_OUT"; then
        die "Stage 1 could not settle on one face. Crop to a single subject, or pass --face-index N."
    elif [[ "$MODE" == web-reverse ]]; then
        _show_candidate_log
        die "NO_RESULTS — the provider answered, and either it found no matching images at all, none of them were on a social platform this pipeline handles, or none of the candidates passed face verification. All three are honest answers about this photo, and none of them is a provider failure (that would have exited 2). Widen the platform map with --social-domain platform:domain if the match is on a platform not in the default list."
    elif (( BY_FACE )) && [[ -n "$CORPUS_DB" ]]; then
        _show_candidate_log
        die "no match above threshold — the corpus database at $CORPUS_DB was searched by face and the probe is not in it. That is the honest outcome of a fixed corpus, not a failure. Add the subject to it and re-run: faceanchor corpus-build --db $CORPUS_DB --from-dir <dir-of-their-photos> --append, or --corpus mastodon:account/<their-handle> --append. 'faceanchor corpus-footprint --probe $PROBE --db $CORPUS_DB' shows every score, so you can see how close it came."
    elif (( BY_FACE )); then
        _show_candidate_log
        die "no match above threshold — the corpus was searched by face, and the probe is not in it. That is the honest outcome of a bounded corpus (PRD §11): images pulled from public hashtag timelines and feeds are strangers. Add a corpus that does contain the subject — rerun with --handle <your handle>, or --corpus mastodon:account/<handle> / --corpus bsky:author/<handle> — and the same by-face search will retrieve them. The log above lists every URL that was searched and why each one lost."
    elif (( CAPTURED )); then
        die "no match above threshold — Stage 2 looked where --query pointed and your face wasn't there. That is the correct answer, not a failure: text-seeded discovery can only find you where words about you already point. Re-run with --query set to your own Mastodon/Bluesky handle, or try --web-reverse, which submits the photo itself to a reverse-image provider instead of searching for words."
    else
        die "no match above threshold. Discovery is corpus-bounded (PRD §11): try a --query that names the subject the way a public post would, or a subject with a Wikimedia Commons portrait."
    fi
fi
[[ -f evidence/latest.json ]] || die "no evidence/latest.json was written"
cp evidence/latest.json evidence/demo-baseline.json
ok "evidence bundle written and anchored"
_show_candidate_log
note "what actually went into it about the subject — hashes, nothing invertible:"
show "python -c \"import json;print(json.dumps(json.load(open('evidence/latest.json'))['probe'],indent=2))\""
"$VENV/python" -c "import json;print(json.dumps(json.load(open('evidence/latest.json'))['probe'],indent=2))"
pause

# ── Act 5b · footprint over the corpus ───────────────────────────────────
if [[ -n "$CORPUS_DB" ]]; then
    act "Footprint" "the same corpus, every score — not just the winning one"
    note "run --by-face stops at the best match, which is the right shape for anchoring and the"
    note "  wrong shape for judging it. This shows the whole distribution: if the top hit stands"
    note "  clear of the runner-up it is worth believing; if it does not, go and look."
    runv $FA corpus-footprint --probe "$PROBE" --db "$CORPUS_DB" --subject-consent "$CONSENT" --top 6 \
        --report evidence/footprint.json --contact-sheet evidence/footprint.html \
        && ok "footprint written (evidence/footprint.json, evidence/footprint.html)" \
        || bad "corpus-footprint failed"
    pause
fi

# ── Act 6 · independent re-verification ──────────────────────────────────
act "Re-verification" "a third party re-derives the digest from the file and asks the chain"
runv $FA verify --bundle evidence/latest.json "${CHAIN_ARGS[@]}" \
    && ok "chain confirms the anchor" || bad "verify failed on an unmodified bundle"
pause

# ── Act 7 · tamper ───────────────────────────────────────────────────────
act "Tamper" "edit one number in the evidence file; the chain notices"
show "python3 -c \"...\"  # evidence/demo-tampered.json: score 0.9999"
"$VENV/python" - <<'PY'
import json
d = json.load(open("evidence/demo-baseline.json"))
before = d["score"]["value"]
d["score"]["value"] = 0.9999
json.dump(d, open("evidence/demo-tampered.json", "w"), indent=2)
print(f"score {before} -> 0.9999 (one field, in one file)")
PY
show "faceanchor verify --bundle evidence/demo-tampered.json"
if $FA verify --bundle evidence/demo-tampered.json "${CHAIN_ARGS[@]}"; then
    bad "the tampered bundle verified — canonicalization or the anchor is broken"
else
    ok "FAIL is the correct result: that digest was never anchored"
fi
pause

# ── Act 8 · ZK match proof (§7.1) ────────────────────────────────────────
if (( DO_ZK )); then
    act "ZK match proof (§7.1)" "prove the score cleared the threshold without publishing either faceprint"
    ZK_ARGS=("${RUN_ARGS[@]}" --zk)
    show "faceanchor run ... --zk   # same discovery mode as act 5, plus the proof"
    if $FA "${ZK_ARGS[@]}"; then
        show "grep -ci 'embedding' evidence/latest.json   # the whole point of the ZK rung"
        EMB_HITS=$(grep -ci embedding evidence/latest.json || true)
        echo "$EMB_HITS"
        if [[ "$EMB_HITS" == "0" ]]; then
            ok "score proven above threshold, with neither faceprint in the bundle"
        else
            bad "bundle contains $EMB_HITS line(s) mentioning \"embedding\" — raw biometric may have leaked"
        fi
        cp evidence/latest.json evidence/demo-zk.json

        # ── Act 9 · forged score ─────────────────────────────────────────
        act "Forged score (§12)" "claim 0.99, reuse the honest proof, and let the chain decide"
        show "faceanchor forge-score --bundle evidence/demo-zk.json --score 0.99"
        FORGE_OUT=$($FA forge-score --bundle evidence/demo-zk.json --score 0.99 "${CHAIN_ARGS[@]}" 2>&1)
        echo "$FORGE_OUT"
        if grep -q 'chain REJECTS' <<<"$FORGE_OUT"; then
            ok "the chain enforces the proof rather than trusting the claimed score"
        else
            bad "expected 'chain REJECTS' from forge-score"
        fi
    else
        bad "the --zk run failed"
        ACT=$((ACT + 1))  # act 8 never ran
    fi
    pause
else
    ACT=$((ACT + 2))  # keep act numbering stable whether or not ZK ran
fi

# ── Act 10 · platform-mutation check (§7.2) ──────────────────────────────
act "Platform-mutation check (§7.2)" "the anchor is honest, but did the post itself change after we saw it?"
runv $FA verify --bundle evidence/demo-baseline.json "${CHAIN_ARGS[@]}" --check-platform \
    && ok "platform check ran" || bad "verify --check-platform failed"
pause

# ── Act 11 · consent revocation (§7.3) ───────────────────────────────────
act "Consent revocation (§7.3)" "the anchor is permanent; the subject's veto over it is too"
show "faceanchor revoke --consent $CONSENT"
if $FA revoke --consent "$CONSENT" "${CHAIN_ARGS[@]}" 2>&1 | tail -4; then
    :
else
    # EvidenceAnchor.AlreadyRevoked (selector 0x90315de1) if this same consent
    # digest was already revoked against a long-lived chain by an earlier run.
    note "revoke reverted — most likely already revoked on this chain by an earlier demo run; the verify below is the assertion that matters"
fi
show "faceanchor verify --bundle evidence/demo-baseline.json ${CHAIN_ARGS[*]}"
VERIFY_OUT=$($FA verify --bundle evidence/demo-baseline.json "${CHAIN_ARGS[@]}" 2>&1)
echo "$VERIFY_OUT"
if grep -q 'CONSENT REVOKED' <<<"$VERIFY_OUT"; then
    ok "still anchored — and now permanently, publicly flagged as revoked"
else
    bad "verify did not surface the revocation"
fi

# ── summary ──────────────────────────────────────────────────────────────
printf '\n%s\n' "$DIM$(printf '─%.0s' {1..72})$R"
printf '%sArtifacts%s  evidence/demo-baseline.json' "$B" "$R"
[[ -f evidence/demo-tampered.json ]] && printf ', demo-tampered.json'
[[ -f evidence/demo-zk.json ]] && printf ', demo-zk.json'
printf '\n%sContract%s   %s on %s\n' "$B" "$R" "$CONTRACT" "$RPC_URL"
if (( CAPTURED )); then
    printf '%sProbe%s      %s — a face image. Gitignored, not anchored, delete it when done.\n' "$B" "$R" "$PROBE"
fi

if ((${#SKIPPED[@]})); then
    printf '\n%sSkipped:%s\n' "$YEL" "$R"
    printf '  ⊘ %s\n' "${SKIPPED[@]}"
fi
if ((${#FAILURES[@]})); then
    printf '\n%s%d act(s) did not behave as the README claims:%s\n' "$RED" "${#FAILURES[@]}" "$R"
    printf '  ✗ %s\n' "${FAILURES[@]}"
    exit 1
fi
printf '\n%s✓ every act behaved exactly as the README claims.%s\n' "$GRN" "$R"
