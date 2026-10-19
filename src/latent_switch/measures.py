"""Per-layer measures for one prompt: matched words vs control words.

`roles` maps a role name to a language, e.g. {"tgt": "cmn_Hans", "lat": "eng_Latn"} or, for the hub test,
{"tgt": "deu_Latn", "en": "eng_Latn", "zh": "cmn_Hans"}. Role names become curve-name prefixes:
`{role}_first`, `{role}_prefix`, `{role}_ctrl_first`, `{role}_ctrl_prefix`, `rank_{role}`, `rank_{role}_ctrl`.
"""

import math

import numpy as np
import pandas as pd
import torch


def make_controls(n: int, seed: int = 0) -> np.ndarray:
    """Derangement: ctrl[i] = row used as control for prompt i (never i itself).

    Every row is used as a control exactly once, so word popularity cancels in averages over all prompts.
    """
    if n < 2:
        raise ValueError("need at least 2 rows to pair each with a different one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)               # a random ordering of 0..n-1
    ctrl = np.empty(n, dtype=int)
    ctrl[order] = np.roll(order, 1)          # pair order[k] with order[k-1] (wraps around)
    return ctrl


def entropy_bits(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over the last dim, in bits; entr() treats 0·log 0 as 0."""
    return torch.special.entr(probs).sum(-1) / math.log(2)


def measure(probs: torch.Tensor, row: pd.Series, ctrl_row: pd.Series, roles: dict[str, str]) -> dict:
    """Per-layer curves (each [n_layers]) for one prompt.

    Matched = this concept's word in each role's language; control = the control concept's word.
    Prefix ids shared between any matched set and any control set are removed from both (they cannot tell
    the words apart). A control whose first token equals a matched first token gets nan.
    """
    n_layers, dev, nan = probs.shape[0], probs.device, float("nan")

    def p_of(ids):
        if not ids:
            return torch.full((n_layers,), nan, device=dev)
        return probs[:, torch.tensor(sorted(ids), device=dev)].sum(-1)

    def rank_of(i):
        if i is None:
            return torch.full((n_layers,), nan, device=dev)
        return (probs > probs[:, [i]]).sum(-1).float()

    sets = {r: set(row[f"{lang}_prefix"]) for r, lang in roles.items()}
    ctrl_sets = {r: set(ctrl_row[f"{lang}_prefix"]) for r, lang in roles.items()}
    shared = set().union(*sets.values()) & set().union(*ctrl_sets.values())
    first = {r: row[f"{lang}_id"] for r, lang in roles.items()}

    out = {}
    for r, lang in roles.items():
        cid = ctrl_row[f"{lang}_id"]
        cid = None if cid in first.values() else cid
        out[f"{r}_first"] = p_of({first[r]})
        out[f"{r}_prefix"] = p_of(sets[r] - shared)
        out[f"{r}_ctrl_first"] = p_of({cid} - {None})
        out[f"{r}_ctrl_prefix"] = p_of(ctrl_sets[r] - shared)
        out[f"rank_{r}"] = rank_of(first[r])
        out[f"rank_{r}_ctrl"] = rank_of(cid)
    out["entropy"] = entropy_bits(probs)
    return out
