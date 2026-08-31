from faceanchor.search.duckduckgo import _image_candidates_from_ddgs_results, extract_seed_terms


def test_extract_seed_terms_uses_handle_and_text():
    seed = extract_seed_terms("alice.bsky.social", "Alice at the conference, great talk!")
    assert "alice.bsky.social" in seed
    assert "conference" in seed


def test_extract_seed_terms_strips_mastodon_instance_suffix():
    seed = extract_seed_terms("bob@mastodon.social", None)
    assert seed == "bob"


def test_extract_seed_terms_drops_stopwords_and_short_tokens():
    seed = extract_seed_terms(None, "the cat and the hat on a mat")
    terms = seed.split()
    assert "the" not in terms
    assert "and" not in terms


def test_extract_seed_terms_respects_max_terms():
    seed = extract_seed_terms(None, "alpha beta gamma delta epsilon zeta eta", max_terms=3)
    assert len(seed.split()) <= 3


def test_extract_seed_terms_empty_input_returns_empty():
    assert extract_seed_terms(None, None) == ""


def test_image_candidates_from_ddgs_results_parses_expected_shape():
    results = [
        {
            "title": "a photo",
            "image": "https://example.com/photo.jpg",
            "url": "https://example.com/page",
            "source": "example.com",
        },
        {"title": "no image field"},  # should be skipped
    ]
    candidates = _image_candidates_from_ddgs_results(results)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.platform == "ddg"
    assert c.image_url == "https://example.com/photo.jpg"
    assert c.post_uri == "https://example.com/page"
