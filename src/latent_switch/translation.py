"""A1 translation-lens experiment: run all prompts, save and load results."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from latent_switch.lens import layer_probs, script_curves
from latent_switch.measures import measure


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


def save_run(run: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(run, path)
    return path


def load_run(path: str | Path) -> dict:
    """weights_only=True (the default) is enough: runs hold only tensors and plain Python types."""
    run = torch.load(path)
    run.setdefault("roles", {"tgt": run["config"]["tgt"], "lat": run["config"]["latent"]})   # pre-roles runs
    return run
