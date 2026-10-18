import pytest
from transformers import AutoTokenizer

from latent_switch.tokens import (
    bytes_to_ids,
    is_whole_chars,
    prefix_ids,
    token_bytes,
    token_stats,
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")


@pytest.fixture(scope="module")
def lookup(tok):
    return bytes_to_ids(tok)


def test_token_bytes_roundtrip(tok):
    tb = token_bytes(tok)
    assert len(tb) == len(tok)
    ids = tok.encode(" cave 云", add_special_tokens=False)
    assert b"".join(tb[i] for i in ids) == " cave 云".encode()


def test_is_whole_chars():
    assert is_whole_chars("云".encode())
    assert not is_whole_chars("云".encode()[:1])


def test_token_stats_space_variant(tok):
    s = token_stats("云", tok)
    assert s["n_tok_plain"] == 1 and s["first_ok_plain"]
    assert not s["first_ok_space"]          # ' 云' starts with a bare space token


def test_prefix_ids(tok, lookup):
    assert [tok.decode([i]) for i in prefix_ids("云", lookup)] == ["云"]
    assert len(prefix_ids("云", lookup, allow_partial=True)) == 3
    assert " cave" in {tok.decode([i]) for i in prefix_ids("cave", lookup)}
