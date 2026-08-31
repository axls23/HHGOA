from faceanchor.chain.solana import _memo_payload


def test_memo_payload_encodes_digest_and_cid():
    digest = bytes.fromhex("aa" * 32)
    payload = _memo_payload(digest, "bafyreitest")
    assert payload == b"faceanchor:" + b"aa" * 32 + b":bafyreitest"


def test_memo_payload_is_deterministic():
    digest = bytes.fromhex("bb" * 32)
    assert _memo_payload(digest, "cid-a") == _memo_payload(digest, "cid-a")
    assert _memo_payload(digest, "cid-a") != _memo_payload(digest, "cid-b")
