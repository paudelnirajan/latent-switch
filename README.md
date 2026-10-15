# latent-switch

When a multilingual LLM reasoning in language X suddenly switches to another language, was that switch
already visible in its middle layers a few tokens earlier, or is it a slip at the output?

The setup is language-agnostic. Nepali is the development language; experiments run over a panel of
languages.

## Setup

```bash
uv sync                                    # creates .venv from pyproject.toml + uv.lock
uv run python -m ipykernel install --user --name latent-switch   # Jupyter kernel (once)
uv run nbstripout --install                # strip notebook outputs from commits (once per clone)
```

Open notebooks with the **latent-switch** kernel.

## Layout

```
src/latent_switch/   reusable code (lens, tokens, languages, plotting); import from notebooks
notebooks/           exploration and debugging, numbered in reading order
scripts/             entry points for long runs and cluster jobs (python scripts/x.py --config ...)
configs/             experiment configs (models, languages, decoding settings)
data/concepts/       small hand-curated inputs (concept lists), tracked in git
data/raw, cache      downloaded / derived data, not tracked
results/             outputs (plots, tensors, CSVs), not tracked
tests/               quick checks for src/ code
external/            third-party repos we read from, not tracked
notes/               private research guide and references, not tracked
```

Rule of thumb: once a function gets copy-pasted into a second notebook, move it into `src/latent_switch/`.
