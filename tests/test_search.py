from faceanchor.search.bluesky import _extract_image_candidates as bsky_extract
from faceanchor.search.fanout import sha256_hex
from faceanchor.search.mastodon import _extract_image_candidates as mastodon_extract
from faceanchor.search.models import Candidate
from faceanchor.vision.phash import hamming_distance


def test_bluesky_extract_image_candidates():
    post = {
        "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
        "cid": "bafyreiabc123",
        "author": {"handle": "alice.bsky.social"},
        "record": {"text": "hello world"},
        "embed": {
            "images": [
                {"fullsize": "https://cdn.bsky.app/img/feed_fullsize/plain/abc/def.jpg", "alt": "a photo"},
            ]
        },
    }
    candidates = bsky_extract(post)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.platform == "bsky"
    assert c.image_url == "https://cdn.bsky.app/img/feed_fullsize/plain/abc/def.jpg"
    assert c.author == "alice.bsky.social"
    assert c.record_cid == "bafyreiabc123"
    assert c.text == "hello world"


def test_bluesky_extract_handles_record_with_media_embed():
    post = {
        "uri": "at://did:plc:abc/app.bsky.feed.post/1",
        "cid": "bafyreidef",
        "author": {"handle": "bob.bsky.social"},
        "record": {"text": "quote post"},
        "embed": {"media": {"images": [{"fullsize": "https://cdn.bsky.app/img/2.jpg"}]}},
    }
    candidates = bsky_extract(post)
    assert len(candidates) == 1
    assert candidates[0].image_url == "https://cdn.bsky.app/img/2.jpg"


def test_bluesky_extract_no_images_returns_empty():
    post = {"uri": "at://x", "cid": "y", "author": {"handle": "z"}, "record": {"text": "no images here"}}
    assert bsky_extract(post) == []


def test_mastodon_extract_image_candidates():
    status = {
        "url": "https://mastodon.social/@alice/12345",
        "account": {"acct": "alice"},
        "content": "<p>a post</p>",
        "media_attachments": [
            {"type": "image", "url": "https://files.mastodon.social/media/1.jpg", "description": "a face"},
            {"type": "video", "url": "https://files.mastodon.social/media/2.mp4"},
        ],
    }
    candidates = mastodon_extract(status)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.platform == "mastodon"
    assert c.image_url == "https://files.mastodon.social/media/1.jpg"
    assert c.author == "alice"
    assert c.extra["description"] == "a face"


def test_mastodon_extract_ignores_non_image_media():
    status = {
        "url": "https://mastodon.social/@bob/1",
        "account": {"acct": "bob"},
        "content": "",
        "media_attachments": [{"type": "audio", "url": "https://files.mastodon.social/media/a.mp3"}],
    }
    assert mastodon_extract(status) == []


def test_sha256_hex_matches_known_vector():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hamming_distance_zero_for_identical_hashes():
    h = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    assert hamming_distance(h, h) == 0


def test_hamming_distance_counts_bit_differences():
    a = bytes([0b00000000])
    b = bytes([0b00000011])
    assert hamming_distance(a, b) == 2
