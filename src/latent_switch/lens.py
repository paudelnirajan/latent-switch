"""Logit lens readouts: per-layer next-token probabilities and script-level token masks."""

import regex
import torch

from latent_switch.languages import UNICODE_SCRIPT
from latent_switch.tokens import is_whole_chars, token_bytes


def layer_probs(model, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ln_final logit lens at the last position, for every block output (resid_post).

    Returns probs [n_layers, d_vocab] (padded vocab columns get probability 0) and the model's own final
    logits [d_vocab]. probs[-1] equals the model's output distribution.
    """
    n_layers, n_real = model.cfg.n_layers, len(model.tokenizer)
    tokens = model.to_tokens(prompt, prepend_bos=False)                     # [1, seq]
    logits, cache = model.run_with_cache(
        tokens, names_filter=lambda name: name.endswith("hook_resid_post"))

    resid = torch.stack([cache["resid_post", L][0, -1] for L in range(n_layers)])   # [n_layers, d_model]
    lens = (model.ln_final(resid) @ model.W_U + model.b_U).float()          # [n_layers, d_vocab]
    lens[:, n_real:] = -torch.inf                                           # padded vocab → prob 0
    return lens.softmax(dim=-1), logits[0, -1].float()


def script_mask(tok, script: str, d_vocab: int) -> torch.Tensor:
    """Bool [d_vocab]: True for tokens whose letters are all in `script` (whole-character tokens, ≥1 letter).

    Special tokens and partial-character byte tokens are excluded. Script ≠ language: Latin covers en, fr, de...
    """
    only_script = regex.compile(rf"^[\p{{{UNICODE_SCRIPT[script]}}}\P{{L}}]*$")   # this script's letters or non-letters
    has_letter = regex.compile(r"\p{L}")
    special = set(tok.added_tokens_decoder)                                       # <|endoftext|> etc.
    mask = torch.zeros(d_vocab, dtype=torch.bool)
    for tid, b in token_bytes(tok).items():
        if tid in special or not is_whole_chars(b):
            continue
        text = b.decode("utf-8")
        if has_letter.search(text) and only_script.match(text):
            mask[tid] = True
    return mask


def script_curves(probs: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """P(any token of each script) per layer: {'script_Hans': [n_layers], ...}."""
    return {f"script_{s}": probs[:, mk].sum(-1) for s, mk in masks.items()}
