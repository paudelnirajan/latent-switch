import pandas as pd
import pytest

from latent_switch.languages import endonym, script_of
from latent_switch.prompts import build_prompts


@pytest.fixture
def df():
    return pd.DataFrame({"fra_Latn": [f"fr{i}" for i in range(8)], "cmn_Hans": [f"zh{i}" for i in range(8)]})


def test_languages():
    assert script_of("npi_Deva") == "Deva"
    assert endonym("cmn_Hans") == "中文"
    with pytest.raises(KeyError):
        endonym("xxx_Latn")


def test_build_prompts(df):
    prompts = build_prompts(df, "fra_Latn", "cmn_Hans", k_shot=5, seed=0)
    assert len(prompts) == len(df)
    assert prompts == build_prompts(df, "fra_Latn", "cmn_Hans", k_shot=5, seed=0)    # reproducible
    for i, p in enumerate(prompts):
        lines = p.split("\n")
        assert len(lines) == 5
        assert lines[-1] == f'Français: "fr{i}" - 中文: "'                              # query last, open quote
        assert all(f'"fr{i}"' not in line for line in lines[:-1])                    # query never a demo
