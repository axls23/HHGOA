"""Drop sexually-explicit and graphic candidates before they are ever fetched.

Discovery over public timelines is untargeted by construction: `--corpus
mastodon:tag/selfie` and `bsky:feed/whats-hot` return whatever strangers
posted, and on both platforms some of that is explicit. Left unfiltered it
reaches three places it must not: a `--contact-sheet` that inlines third-party
images into an HTML file on disk, a `--log-candidates` file naming their URLs,
and — if one ever scored above threshold — an evidence bundle.

The filter runs at *extraction*, not after download, so an excluded post costs
no request, no bytes and no decode. Nothing explicit is fetched at all.

Both platforms publish the metadata already; this module only reads it:

- **Mastodon** — `status.sensitive` is the poster's own content-warning flag,
  set through the client's CW control and routinely enforced by instance
  rules. `spoiler_text` is the warning's text. Instances that host adult
  content generally require it, which is what makes the flag worth trusting
  as a *first* line rather than the only one.
- **Bluesky** — AT Protocol self-labels and labeler labels arrive on the post
  view as `labels[].val` (`porn`, `sexual`, `nudity`, `graphic-media`, …).
  The author's own account labels are checked too, since accounts that post
  adult content routinely label the account rather than each post.

What this is **not**: a classifier, and not a guarantee. It trusts the
poster and their instance's moderation. An unflagged explicit post is not
caught here — see `docs/CONSENT.md` and the README's "Filtering explicit
content" section for what that means and what to do about it. That residual
risk is the reason the demo defaults to hashtags chosen to be low-risk, and
the reason nothing is fetched from an instance on `ADULT_INSTANCES`.

Filtering is on by default and reported per source (`filtered_sensitive`),
never silent — the same posture the rest of `search/` takes towards anything
that changes recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# AT Protocol label values that mean sexual or graphic content. `nsfw`,
# `nudity-nonsexual` and `gore` are legacy/self-label spellings still seen in
# the wild alongside the current moderation vocabulary.
BSKY_EXPLICIT_LABELS = frozenset(
    {
        "porn", "sexual", "nudity", "sexual-figurative", "nsfw",
        "graphic-media", "gore", "self-harm", "nudity-nonsexual",
    }
)

# Hashtags that are adult corpora by definition. Refusing these outright is
# not prudishness about the tag — it is that ingesting one means downloading
# hundreds of strangers' explicit images to search for a face, which no
# consent artifact covers and no demo needs.
ADULT_TAGS = frozenset(
    {
        "nsfw", "porn", "porno", "pornography", "nude", "nudes", "nudity",
        "sex", "sexy", "erotic", "erotica", "hentai", "boobs", "onlyfans",
        "lewd", "explicit", "xxx", "adultcontent", "nsfwart", "nsfwtwitter",
    }
)

# Instances whose entire purpose is adult content. A per-post flag is the
# wrong tool here: the right answer is not to page through the timeline at
# all. Not exhaustive, and not meant to be — it is a floor, not a wall.
ADULT_INSTANCES = frozenset(
    {
        "baraag.net", "pawoo.net", "bahamut.social", "sinblr.com",
        "switter.at", "humblr.social", "kinkyelephant.com", "kinky.business",
        "rubber.social", "mstdn.nsfw.social", "nsfw.social",
    }
)


class AdultCorpusError(ValueError):
    """A corpus spec that names an adult tag or instance outright."""


@dataclass
class SafetyReport:
    """How many candidates the filter removed, and why.

    Kept per-run and printed, because a filter that silently halves recall is
    indistinguishable from a search that found nothing — the same reason
    `SourceReport` exists.
    """

    sensitive: int = 0
    labelled: int = 0
    by_label: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.sensitive + self.labelled

    def note_sensitive(self) -> None:
        self.sensitive += 1

    def note_labelled(self, labels: list[str]) -> None:
        self.labelled += 1
        for label in labels:
            self.by_label[label] = self.by_label.get(label, 0) + 1

    def describe(self) -> str:
        if not self.total:
            return "none"
        parts = []
        if self.sensitive:
            parts.append(f"{self.sensitive} flagged sensitive (mastodon)")
        if self.labelled:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(self.by_label.items()))
            parts.append(f"{self.labelled} labelled (bsky: {detail})")
        return "; ".join(parts)


def mastodon_status_is_sensitive(status: dict) -> bool:
    """True when the poster or their instance flagged this status.

    `sensitive` is the flag proper. A non-empty `spoiler_text` is also treated
    as a warning, because some clients set the CW text without the boolean and
    a content warning is a content warning either way.
    """
    if status.get("sensitive") is True:
        return True
    return bool((status.get("spoiler_text") or "").strip())


def bsky_explicit_labels(post: dict) -> list[str]:
    """Explicit label values on a post view, its record, or its author.

    Three places because the platform uses all three: a labeler labels the
    post, an author self-labels the record at write time, and an adult account
    is usually labelled once at the account level rather than per post.
    """
    found: set[str] = set()

    def _collect(entries) -> None:
        for entry in entries or []:
            if isinstance(entry, dict):
                value = entry.get("val")
            elif isinstance(entry, str):
                value = entry
            else:
                continue
            if value and value.lower() in BSKY_EXPLICIT_LABELS:
                found.add(value.lower())

    _collect(post.get("labels"))
    record = post.get("record") or {}
    _collect(record.get("labels"))
    _collect((record.get("labels") or {}).get("values") if isinstance(record.get("labels"), dict) else None)
    _collect((post.get("author") or {}).get("labels"))
    return sorted(found)


def assert_corpus_is_not_adult(platform: str, kind: str, value: str, instance: str | None) -> None:
    """Refuse a corpus spec that names an adult tag or instance up front.

    Raised at spec-parse time, before the first request — the point is that
    the run never happens, not that its results are filtered afterwards.
    """
    if instance and instance.lower() in ADULT_INSTANCES:
        raise AdultCorpusError(
            f"refusing to ingest {instance!r}: it is an adult-content instance. "
            "Paging its timeline means downloading strangers' explicit images to search for a face, "
            "which no consent artifact covers."
        )
    if platform == "mastodon" and kind == "tag" and value.lower().lstrip("#") in ADULT_TAGS:
        raise AdultCorpusError(
            f"refusing to ingest #{value}: it is an adult-content tag. "
            "Pick a corpus that does not require downloading explicit images of strangers."
        )
