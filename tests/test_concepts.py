import pandas as pd

from latent_switch.concepts import translation_set

F, D, E, Z = "fra_Latn", "deu_Latn", "eng_Latn", "cmn_Hans"


def _stats(rows):
    """Minimal stats table: (concept, lang, prefix ids); first id = first prefix id."""
    return pd.DataFrame([{"concept_id": c, "lang": l, "word": f"{c}_{l}", "first_id_plain": ids[0],
                          "first_ok_plain": True, "prefix_ids": ids} for c, l, ids in rows])


def test_translation_set_multi_latent_drops_shared_prefixes():
    stats = _stats([
        ("A", F, [1]), ("A", D, [2]), ("A", E, [3]), ("A", Z, [4]),         # all disjoint → kept
        ("B", F, [5]), ("B", D, [6, 9]), ("B", E, [7]), ("B", Z, [8, 9]),   # de and zh share 9 → dropped
    ])
    out = translation_set(stats, F, D, [E, Z])
    assert out["concept_id"].tolist() == ["A"]
    assert {f"{l}_prefix" for l in (D, E, Z)} <= set(out.columns)
    assert len(translation_set(stats, F, D, E)) == 2       # single latent: B's clash with zh is irrelevant
