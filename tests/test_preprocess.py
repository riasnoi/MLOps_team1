import pytest

from src import preprocess as prep


def test_clean_text_normalizes_html_and_urls():
    raw = "Call <b>NOW</b> at https://Example.com/path?A=1\nNew line"
    cleaned = prep.clean_text(raw)
    assert cleaned == "call now at <url> new line"


def test_num_domains_counts_unique_domains():
    text = (
        "Visit https://one.example.com/offer and "
        "https://two.example.net/deal and "
        "http://one.example.com/more"
    )
    assert prep.num_domains(text) == 2


def test_upper_ratio_handles_mixed_case():
    assert prep.upper_ratio("ABcd") == pytest.approx(0.5)
