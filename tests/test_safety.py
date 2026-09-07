"""Explicit-content filtering at the discovery layer.

The harm this prevents is concrete: a `--corpus mastodon:tag/...` run over a
public timeline pulls in whatever strangers posted, and unfiltered that
reaches a `--contact-sheet` (which inlines third-party images into an HTML
file on disk), a `--log-candidates` file naming their URLs, and potentially
an anchored evidence bundle. So the tests assert the filter runs *before the
fetch*, not after.
"""

from __future__ import annotations

import pytest

from faceanchor.search.bluesky import _avatar_candidate as bsky_avatar
from faceanchor.search.bluesky import _extract_image_candidates as bsky_extract
from faceanchor.search.mastodon import _extract_image_candidates as mastodon_extract
from faceanchor.search.safety import (
    ADULT_INSTANCES,
    ADULT_TAGS,
    AdultCorpusError,
    SafetyReport,
    assert_corpus_is_not_adult,
    bsky_explicit_labels,
    mastodon_status_is_sensitive,
)


def _masto_status(**overrides) -> dict:
    status = {
        "url": "https://mastodon.social/@alice/12345",
        "account": {"acct": "alice"},
        "content": "<p>a post</p>",
        "media_attachments": [{"type": "image", "url": "https://files.mastodon.social/media/1.jpg"}],
    }
    status.update(overrides)
    return status


def _bsky_post(**overrides) -> dict:
    post = {
        "uri": "at://did:plc:abc/app.bsky.feed.post/1",
        "cid": "bafyreiabc",
        "author": {"handle": "alice.bsky.social"},
        "record": {"text": "hello"},
        "embed": {"images": [{"fullsize": "https://cdn.bsky.app/img/1.jpg"}]},
    }
    post.update(overrides)
    return post


# ── Mastodon: the poster's own content warning ───────────────────────────


def test_mastodon_sensitive_status_yields_no_candidates():
    assert mastodon_extract(_masto_status(sensitive=True)) == []


def test_mastodon_content_warning_text_counts_as_a_warning():
    """Some clients set the CW text without the boolean. A content warning is
    a content warning either way."""
    assert mastodon_extract(_masto_status(spoiler_text="nsfw")) == []
    assert mastodon_status_is_sensitive({"spoiler_text": "eye contact"}) is True
    assert mastodon_status_is_sensitive({"spoiler_text": "   "}) is False


def test_mastodon_unflagged_status_is_unaffected():
    """The default path must not change: an ordinary post still yields its images."""
    candidates = mastodon_extract(_masto_status())
    assert len(candidates) == 1
    assert candidates[0].image_url == "https://files.mastodon.social/media/1.jpg"

    assert mastodon_status_is_sensitive({"sensitive": False}) is False
    assert mastodon_status_is_sensitive({}) is False


def test_mastodon_filter_is_opt_out_not_opt_in():
    assert mastodon_extract(_masto_status(sensitive=True), allow_sensitive=True) != []


def test_mastodon_filter_counts_what_it_removed():
    report = SafetyReport()
    mastodon_extract(_masto_status(sensitive=True), report=report)
    mastodon_extract(_masto_status(sensitive=True), report=report)
    mastodon_extract(_masto_status(), report=report)
    assert report.sensitive == 2
    assert report.total == 2
    assert "2 flagged sensitive" in report.describe()


# ── Bluesky: AT Protocol labels ──────────────────────────────────────────


@pytest.mark.parametrize("label", ["porn", "sexual", "nudity", "graphic-media", "nsfw", "gore"])
def test_bsky_labelled_post_yields_no_candidates(label):
    assert bsky_extract(_bsky_post(labels=[{"val": label, "src": "did:plc:labeler"}])) == []


def test_bsky_reads_labels_from_post_record_and_author():
    """The platform uses all three placements, so all three are checked."""
    assert bsky_explicit_labels({"labels": [{"val": "porn"}]}) == ["porn"]
    assert bsky_explicit_labels({"record": {"labels": {"values": [{"val": "sexual"}]}}}) == ["sexual"]
    assert bsky_explicit_labels({"author": {"labels": [{"val": "nudity"}]}}) == ["nudity"]
    assert bsky_explicit_labels({"labels": [{"val": "!warn"}, {"val": "spam"}]}) == []
    assert bsky_explicit_labels({}) == []


def test_bsky_account_level_label_also_drops_the_avatar():
    """Adult accounts are usually labelled once, at the account level — the
    avatar route has to honour that too."""
    profile = {
        "handle": "x.bsky.social", "did": "did:plc:x",
        "avatar": "https://cdn.bsky.app/img/avatar/x.jpg",
        "labels": [{"val": "porn"}],
    }
    assert bsky_avatar(profile) == []
    assert bsky_avatar(profile, allow_sensitive=True) != []


def test_bsky_unlabelled_post_is_unaffected():
    candidates = bsky_extract(_bsky_post())
    assert len(candidates) == 1
    assert candidates[0].image_url == "https://cdn.bsky.app/img/1.jpg"


def test_bsky_filter_counts_labels_it_removed():
    report = SafetyReport()
    bsky_extract(_bsky_post(labels=[{"val": "porn"}]), report=report)
    bsky_extract(_bsky_post(labels=[{"val": "porn"}, {"val": "nudity"}]), report=report)
    assert report.labelled == 2
    assert report.by_label == {"porn": 2, "nudity": 1}
    assert "bsky:" in report.describe()


# ── corpus specs refused up front ────────────────────────────────────────


def test_adult_tag_corpus_is_refused_before_any_request():
    from faceanchor.search.corpus import CorpusSpecError, parse_spec

    for tag in ("nsfw", "porn", "hentai", "NSFW", "#nsfw"):
        with pytest.raises(CorpusSpecError, match="adult-content tag"):
            parse_spec(f"mastodon:tag/{tag}")


def test_adult_instance_corpus_is_refused_before_any_request():
    from faceanchor.search.corpus import CorpusSpecError, parse_spec

    with pytest.raises(CorpusSpecError, match="adult-content instance"):
        parse_spec("mastodon:tag/art@baraag.net")


def test_ordinary_corpus_specs_still_parse():
    from faceanchor.search.corpus import parse_spec

    assert parse_spec("mastodon:tag/selfie").value == "selfie"
    assert parse_spec("mastodon:tag/portrait@mstdn.social").instance == "mstdn.social"
    assert parse_spec("bsky:feed/whats-hot").value == "whats-hot"


def test_assert_corpus_is_not_adult_is_case_insensitive():
    with pytest.raises(AdultCorpusError):
        assert_corpus_is_not_adult("mastodon", "tag", "PORN", None)
    with pytest.raises(AdultCorpusError):
        assert_corpus_is_not_adult("mastodon", "tag", "x", "BARAAG.NET")
    assert_corpus_is_not_adult("mastodon", "tag", "selfie", "mastodon.social")  # no raise


def test_denylists_are_non_empty_and_lowercase():
    """A denylist that silently became empty would disable the gate."""
    assert ADULT_TAGS and ADULT_INSTANCES
    assert all(t == t.lower() for t in ADULT_TAGS)
    assert all(i == i.lower() for i in ADULT_INSTANCES)


# ── the reverse-image arm has no post metadata, so it blocks by host ─────


def test_reverse_image_results_from_adult_instances_are_dropped():
    from faceanchor.search.web_reverse import ReverseImageResult, filter_social

    report = filter_social(
        [
            ReverseImageResult("p", "https://baraag.net/media/x.jpg", "https://baraag.net/@a/1", "full"),
            ReverseImageResult("p", "https://cdn.bsky.app/img/ok.jpg", "https://bsky.app/profile/a/post/1", "full"),
        ]
    )
    assert [c.platform for c in report.candidates] == ["bsky"]
    assert [e["reason"] for e in report.skipped] == ["adult_instance"]


# ── nothing explicit is ever fetched ─────────────────────────────────────


async def test_filtered_posts_are_never_fetched():
    """The whole point of filtering at extraction: an excluded post costs no
    request, so its bytes never touch the disk, the log, or a contact sheet."""
    import httpx

    from faceanchor.search.fanout import fetch_all_candidates

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b"bytes")

    candidates = mastodon_extract(_masto_status(sensitive=True)) + mastodon_extract(_masto_status())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fetch_all_candidates(client, candidates)

    assert requested == ["https://files.mastodon.social/media/1.jpg"]
    assert len(requested) == 1
