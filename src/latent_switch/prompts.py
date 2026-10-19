"""Few-shot prompts for the translation lens (Wendler et al. 2024 template)."""

import numpy as np
import pandas as pd

from latent_switch.languages import endonym


def build_prompts(df: pd.DataFrame, src: str, tgt: str, k_shot: int = 5, seed: int = 0) -> list[str]:
    """One k-shot prompt per row of `df`: k-1 demonstrations drawn (seeded) from the other rows, then the query.

    Each line is `{src name}: "{word}" - {tgt name}: "{word}"`; the query line ends on the open quote, so the
    next token is the plain (no leading space) first token of the target word.
    """
    rng = np.random.default_rng(seed)
    a, b = endonym(src), endonym(tgt)
    prompts = []
    for i in range(len(df)):
        others = np.delete(np.arange(len(df)), i)                  # every row except the query
        demo_rows = rng.choice(others, size=k_shot - 1, replace=False)
        lines = [f'{a}: "{df.at[j, src]}" - {b}: "{df.at[j, tgt]}"' for j in demo_rows]
        lines.append(f'{a}: "{df.at[i, src]}" - {b}: "')           # query: open quote, no answer
        prompts.append("\n".join(lines))
    return prompts
