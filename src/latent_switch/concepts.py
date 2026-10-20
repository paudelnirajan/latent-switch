"""Parallel concept table (long format: one row per concept × language) and per-model concept selection."""

from pathlib import Path

import pandas as pd

from latent_switch.tokens import bytes_to_ids, prefix_ids, token_stats

# Wendler et al. 2024 word lists (external/llm-latent-language/data/langs/{short}/clean.csv)
WENDLER_LANGS = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "ru": "rus_Cyrl", "zh": "cmn_Hans"}


def load_wendler(data_dir: str | Path) -> pd.DataFrame:
    """Wendler's clean.csv files as a long table: concept_id, lang, word, source."""
    frames = []
    for short, code in WENDLER_LANGS.items():
        df = pd.read_csv(Path(data_dir) / short / "clean.csv")
        frames.append(pd.DataFrame({
            "concept_id": df["word_original"].str.strip().str.upper(),
            "lang": code,
            "word": df["word_translation"].str.strip(),
            "source": "wendler2024",
        }))
    return pd.concat(frames, ignore_index=True)


def drop_ambiguous(concepts: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose word is used for more than one concept in the same language (e.g. de Boden)."""
    ambiguous = concepts.duplicated(["lang", "word"], keep=False)
    return concepts[~ambiguous].copy()


def add_token_info(concepts: pd.DataFrame, tok, allow_partial: bool = False) -> pd.DataFrame:
    """Add token_stats columns plus `single`, `prefix_ids`, `n_prefix` for one tokenizer."""
    stats = concepts.join(concepts["word"].apply(lambda w: pd.Series(token_stats(w, tok))))
    stats["single"] = (stats["n_tok_plain"] == 1) | (stats["n_tok_space"] == 1)
    lookup = bytes_to_ids(tok)
    stats["prefix_ids"] = stats["word"].apply(lambda w: prefix_ids(w, lookup, allow_partial))
    stats["n_prefix"] = stats["prefix_ids"].apply(len)
    return stats


def select_concepts(stats: pd.DataFrame, langs: list[str], variant: str = "plain", verbose: bool = True):
    """Concepts present in every lang in `langs`, with usable and pairwise-distinct first tokens.

    Returns (wide table with one row per kept concept, index of dropped concepts).
    """
    id_col, ok_col = f"first_id_{variant}", f"first_ok_{variant}"
    wide = stats.pivot(index="concept_id", columns="lang", values=["word", id_col, ok_col])
    wide = wide.dropna(subset=[("word", l) for l in langs])
    n_all = len(wide)

    keep = pd.Series(True, index=wide.index)
    for l in langs:
        keep &= wide[(ok_col, l)].astype(bool)
    for i, a in enumerate(langs):
        for b in langs[i + 1:]:
            keep &= wide[(id_col, a)] != wide[(id_col, b)]

    out = pd.DataFrame(index=wide.index[keep])
    for l in langs:
        out[l] = wide.loc[keep, ("word", l)]
        out[f"{l}_id"] = wide.loc[keep, (id_col, l)].astype(int)
    if verbose:
        print(f"{' / '.join(langs)}: {n_all} concepts in all languages → {len(out)} kept")
    return out.reset_index(), wide.index[~keep]


def translation_set(stats: pd.DataFrame, src: str, tgt: str,
                    latents: str | list[str] = "eng_Latn") -> pd.DataFrame:
    """Concepts for a src→tgt translation lens experiment, with the token sets we measure.

    `latents` is one language or several candidate latent languages (hub test). Filters: first tokens usable
    and distinct across all languages, and the prefix sets of tgt and every latent pairwise disjoint
    (Wendler's rule, generalised). Columns: concept_id, one word + first-id column per language, and
    `{lang}_prefix` (list of ids) for tgt and each latent.
    """
    if isinstance(latents, str):
        latents = [latents]
    measured = list(dict.fromkeys([tgt, *latents]))     # languages we measure (dedupe, keep order)
    langs = list(dict.fromkeys([src, *measured]))
    out, _ = select_concepts(stats, langs, verbose=False)
    prefixes = stats.pivot(index="concept_id", columns="lang", values="prefix_ids")
    for l in measured:
        out[f"{l}_prefix"] = out["concept_id"].map(prefixes[l])

    def disjoint(r):
        sets = [set(r[f"{l}_prefix"]) for l in measured]
        return all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])

    out = out[out.apply(disjoint, axis=1)].reset_index(drop=True)
    print(f"{src} → {tgt} (latents {', '.join(latents)}): {len(out)} concepts")
    return out
