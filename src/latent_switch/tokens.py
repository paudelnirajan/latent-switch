"""Tokenizer helpers that work for any language: token bytes, first tokens, prefix token sets.

Byte-level BPE tokenizers (GPT-2, Qwen) store each token as a string over a 256-character alphabet that
stands for raw bytes. We map tokens back to their real UTF-8 bytes so that "does this token start word W?"
becomes a byte-prefix test, which is the same for every script.
"""

from collections import defaultdict


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen map: each of the 256 bytes is shown as one printable character (e.g. space → 'Ġ')."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_DECODER = {c: b for b, c in bytes_to_unicode().items()}


def token_bytes(tok) -> dict[int, bytes]:
    """{token_id: the raw bytes that token stands for}. Tokens outside the byte alphabet are skipped."""
    out = {}
    for tid in range(len(tok)):
        piece = tok.convert_ids_to_tokens(tid)
        try:
            out[tid] = bytes(_BYTE_DECODER[c] for c in piece)
        except KeyError:
            pass
    return out


def bytes_to_ids(tok) -> dict[bytes, list[int]]:
    """Reverse lookup: bytes → ids of the tokens with exactly those bytes."""
    lookup = defaultdict(list)
    for tid, b in token_bytes(tok).items():
        lookup[b].append(tid)
    return dict(lookup)


def is_whole_chars(b: bytes) -> bool:
    """True if the bytes form complete UTF-8 characters (no half-character at the end)."""
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def token_stats(word: str, tok) -> dict:
    """Token count and first token of `word`, plain and with a leading space."""
    out = {}
    for variant, text in [("plain", word), ("space", " " + word)]:
        ids = tok.encode(text, add_special_tokens=False)
        first = tok.decode(ids[:1])
        out[f"n_tok_{variant}"] = len(ids)
        out[f"first_id_{variant}"] = ids[0]
        out[f"first_str_{variant}"] = first
        # usable first token: not a broken byte piece (shown as �) and not just whitespace
        out[f"first_ok_{variant}"] = ("�" not in first) and first.strip() != ""
    return out


def prefix_ids(word: str, lookup: dict[bytes, list[int]], allow_partial: bool = False) -> list[int]:
    """Every token id whose bytes are a prefix of `word` or of ' ' + `word` (Wendler's token set).

    allow_partial=False skips tokens that end mid-character (e.g. b'\\xe4', which starts thousands of
    CJK characters). Use True to reproduce Wendler's byte-token trick.
    """
    ids = set()
    for text in (word, " " + word):
        b = text.encode("utf-8")
        for k in range(1, len(b) + 1):
            p = b[:k]
            if p.strip() == b"":
                continue
            if not allow_partial and not is_whole_chars(p):
                continue
            ids.update(lookup.get(p, []))
    return sorted(ids)
