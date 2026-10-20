"""A1 translation-lens experiment: run all prompts, save and load results."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from latent_switch.lens import layer_probs, script_curves
from latent_switch.measures import make_controls, measure


def run_translation_lens(model, df: pd.DataFrame, prompts: list[str], ctrl_idx: np.ndarray,
                         roles: dict[str, str], masks: dict[str, torch.Tensor]) -> dict:
    """Run every prompt; `roles` must contain "tgt" (the language being produced).

    Returns a run dict: curves {name: [n_prompts, n_layers]}, final-layer top1 ids, correct (top1 == target
    first token), and the inputs needed to reproduce it (concept ids, control pairing, prompts, roles, scripts).
    """
    curves, top1 = {}, []
    for i in tqdm(range(len(df))):
        probs, _ = layer_probs(model, prompts[i])
        m = measure(probs, df.loc[i], df.loc[ctrl_idx[i]], roles) | script_curves(probs, masks)
        for name, curve in m.items():
            curves.setdefault(name, []).append(curve.cpu())
        top1.append(probs[-1].argmax().item())
    return {
        "curves": {name: torch.stack(cs) for name, cs in curves.items()},
        "top1": top1,
        "correct": (np.array(top1) == df[f"{roles['tgt']}_id"].to_numpy()).tolist(),
        "concept_id": df["concept_id"].tolist(),
        "ctrl_idx": [int(j) for j in ctrl_idx],
        "prompts": prompts,
        "roles": roles,
        "scripts": list(masks),
    }



def run_correct_only(model, df: pd.DataFrame, prompts: list[str], roles: dict[str, str],
                     masks: dict[str, torch.Tensor], seed: int = 0) -> dict:
    """Two passes: find the correctly translated prompts, then rerun only those with controls deranged among them.

    Deranging within the analysed subset keeps the balance make_controls gives: matched and control words are
    the same set of concepts, so word popularity still cancels (deranging over all prompts and then keeping the
    correct ones does not guarantee that).
    """
    first = run_translation_lens(model, df, prompts, make_controls(len(df), seed), roles, masks)
    ok = np.flatnonzero(first["correct"])
    sub = df.iloc[ok].reset_index(drop=True)
    return run_translation_lens(model, sub, [prompts[i] for i in ok], make_controls(len(sub), seed), roles, masks)

def save_run(run: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(run, path)
    return path


def load_run(path: str | Path) -> dict:
    """weights_only=True (the default) is enough: runs hold only tensors and plain Python types."""
    run = torch.load(path)
    if "roles" not in run:                                   # runs saved before roles existed
        run["roles"] = {"tgt": run["config"]["tgt"], "lat": run["config"]["latent"]}
    return run
