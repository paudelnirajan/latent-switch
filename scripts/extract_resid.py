"""Teacher-forced residual-stream activations at chosen positions of the C1 traces.

For every trace in the positions CSV (made in notebooks/07_c1_traces.ipynb, step 5), run ONE forward pass over the
model's own tokens (prompt_ids + gen_ids, up to the last position we need) and keep the hidden states at that trace's
positions, for a subset of layers. No generation, no sampling: the activations are the ones the model had while
writing the trace.

Layer convention (same as notebook 06): layer L = hidden_states[L] = output of block L-1; 0 = embeddings;
the last one (36 for Qwen3-4B) is AFTER the final norm (that is how HF returns it).

Output: a directory of shards, each a dict
    rows   : LongTensor [n]            row numbers in the positions CSV (row i of the CSV <-> one vector)
    layers : list[int]
    resid  : bf16 tensor [n, n_layers, d_model]
    agree  : BoolTensor [n]            does argmax of the next-token logits = the token the model actually wrote?
Shards are written every --shard traces and skipped if they already exist, so a killed job can be resubmitted.
Load everything with load_resid(out_dir).

Examples
  # local smoke test: 3 traces on mps
  uv run python scripts/extract_resid.py --limit 3 --out data/activations/smoke_resid
  # cluster
  python scripts/extract_resid.py --model Qwen/Qwen3-4B
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
KEY = ["source", "item", "lang", "cond", "sample"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--gens", default=None, help="default: data/generations/c1_traces_{tag}.jsonl")
    p.add_argument("--positions", default=None, help="default: data/activations/c1_positions_{tag}.csv")
    p.add_argument("--out", default=None, help="shard directory; default: data/activations/c1_resid_{tag}")
    p.add_argument("--layers", default="12:37:2", help="start:stop:step (python range) or comma list")
    p.add_argument("--shard", type=int, default=250, help="traces per shard file")
    p.add_argument("--limit", type=int, default=None, help="only the first N traces (smoke tests)")
    return p.parse_args()


def parse_layers(s):
    if ":" in s:
        return list(range(*map(int, s.split(":"))))
    return [int(x) for x in s.split(",")]


def load_ids(path, keys):
    """key -> (prompt_ids, gen_ids), only for the keys we need (the jsonl is ~190 MB)."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = tuple(r[c] for c in KEY)
            if k in keys:
                out[k] = (r["prompt_ids"], r["gen_ids"])
    return out


def load_model(name, device):
    import transformers
    from transformers import AutoModelForCausalLM

    # transformers >= 4.56 renamed torch_dtype -> dtype; the cluster's vLLM venv has 4.51
    major, minor = map(int, transformers.__version__.split(".")[:2])
    kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(name, **{kw: torch.bfloat16}).to(device).eval()
    return model


@torch.inference_mode()
def run_trace(model, seq, pos, layers, device):
    """One forward pass. seq: list[int] (already cut after the last needed position); pos: list[int].
    Returns resid [len(pos), len(layers), d] (bf16, cpu) and agree [len(pos)] (bool)."""
    x = torch.tensor([seq], device=device)
    out = model.model(input_ids=x, output_hidden_states=True)      # base model only: no logits over the full seq
    hs = out.hidden_states                                          # tuple of n_layers+1 tensors [1, T, d]
    p = torch.tensor(pos, device=device)
    resid = torch.stack([hs[L][0, p] for L in layers], dim=1)      # [len(pos), len(layers), d]
    # sanity: next-token prediction at the chosen positions only (hs[-1] is already normed)
    logits = model.lm_head(hs[-1][0, p])                            # [len(pos), vocab]
    nxt = torch.tensor([seq[i + 1] if i + 1 < len(seq) else -1 for i in pos], device=device)
    agree = logits.argmax(-1) == nxt
    return resid.cpu(), agree.cpu()


def load_resid(out_dir):
    """Concatenate all shards. Returns rows [N], layers, resid [N, n_layers, d], agree [N], sorted by CSV row."""
    shards = [torch.load(f) for f in sorted(Path(out_dir).glob("shard_*.pt"))]
    rows = torch.cat([s["rows"] for s in shards])
    order = rows.argsort()
    return (rows[order], shards[0]["layers"], torch.cat([s["resid"] for s in shards])[order],
            torch.cat([s["agree"] for s in shards])[order])


def main():
    args = parse_args()
    tag = args.model.split("/")[-1].lower()
    gens = Path(args.gens or REPO / f"data/generations/c1_traces_{tag}.jsonl")
    positions = Path(args.positions or REPO / f"data/activations/c1_positions_{tag}.csv")
    out_dir = Path(args.out or REPO / f"data/activations/c1_resid_{tag}")
    layers = parse_layers(args.layers)

    pos_df = pd.read_csv(positions)
    pos_df["row"] = range(len(pos_df))                              # row id = line in the CSV
    groups = list(pos_df.groupby(KEY, sort=False))                  # one group per trace, CSV order
    if args.limit:
        groups = groups[:args.limit]
    n_shards = (len(groups) + args.shard - 1) // args.shard
    print(f"{len(groups)} traces, {sum(len(g) for _, g in groups)} positions, layers {layers} -> {out_dir} "
          f"({n_shards} shards)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in range(n_shards) if not (out_dir / f"shard_{s:03d}.pt").exists()]
    if not todo:
        print("all shards exist; nothing to do")
        return

    ids = load_ids(gens, {k for k, _ in groups})
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    assert max(layers) <= n_layers, f"model has hidden_states 0..{n_layers}"
    print(f"{device}, {n_layers} blocks, d_model {model.config.hidden_size}", flush=True)

    for s in todo:
        t0, rows, resid, agree, n_tok = time.perf_counter(), [], [], [], 0
        for key, g in groups[s * args.shard:(s + 1) * args.shard]:
            p, gen = ids[key]
            seq = (p + gen)[:g["pos"].max() + 2]                    # +1 for the last position, +1 for its next token
            r, a = run_trace(model, seq, g["pos"].tolist(), layers, device)
            rows.append(torch.tensor(g["row"].to_numpy()))
            resid.append(r)
            agree.append(a)
            n_tok += len(seq)
        rows, resid, agree = torch.cat(rows), torch.cat(resid), torch.cat(agree)
        torch.save(dict(rows=rows, layers=layers, resid=resid, agree=agree), out_dir / f"shard_{s:03d}.pt")
        secs = time.perf_counter() - t0
        print(f"shard {s + 1}/{n_shards}: {len(rows)} positions, {n_tok} tokens, {secs:.0f}s "
              f"({n_tok / secs:.0f} tok/s), next-token agreement {agree.float().mean():.2f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
