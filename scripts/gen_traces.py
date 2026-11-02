"""Generate thinking-on reasoning traces for the C1 prompt set.

One output line per (source, item, lang, cond, sample): token ids of the prompt and of the model's own
generation, plus the decoded trace / answer and a few quality stats. The token ids are what later
teacher-forced activation runs feed back in (never re-tokenize the text: it can split differently).

Conditions
  natural : plain prompt, thinking on
  prefill : "<think>\\n" + an opener written in the prompt language (TRACE_LANGS[lang]["opener"])

Backends
  hf   : transformers generate, one sequence at a time; for local smoke tests (mps/cpu/cuda)
  vllm : batched; for the cluster. Needs its own venv (vLLM pins its own torch/transformers).

The script appends to the output file after every chunk and skips keys already in it, so a job that
hits its time limit or gets preempted can simply be resubmitted.

Examples
  # local smoke test (1.7B, 2 prompts, short)
  uv run python scripts/gen_traces.py --backend hf --model Qwen/Qwen3-1.7B --limit 2 --n-natural 1 \\
      --n-prefill 1 --max-new-tokens 128 --out data/generations/smoke_qwen3-1.7b.jsonl
  # see what would run, without loading a model
  python scripts/gen_traces.py --dry-run
  # cluster
  python scripts/gen_traces.py --backend vllm --model Qwen/Qwen3-4B
"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # import the registry without installing the package (vLLM venv)
from latent_switch.languages import TRACE_LANGS  # noqa: E402

# Qwen3's recommended sampling for thinking mode (greedy decoding loops in thinking mode)
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--prompts", default=str(REPO / "data/prompts/c1_traces.csv"))
    p.add_argument("--out", default=None, help="default: data/generations/c1_traces_{tag}.jsonl")
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    p.add_argument("--conds", default="natural,prefill")
    p.add_argument("--n-natural", type=int, default=8, help="samples per prompt, natural condition")
    p.add_argument("--n-prefill", type=int, default=4, help="samples per prompt, prefill condition")
    p.add_argument("--langs", default=None, help="comma list, e.g. ne,hi,ru (default: all in the CSV)")
    p.add_argument("--sources", default=None, help="comma list: mclm_math100,lcb_dolly (default: both)")
    p.add_argument("--limit", type=int, default=None, help="only the first N prompt rows (smoke tests)")
    p.add_argument("--max-new-tokens", type=int, default=12288)
    p.add_argument("--max-model-len", type=int, default=16384, help="vllm: context length to reserve")
    p.add_argument("--chunk", type=int, default=512, help="requests per write to disk")
    p.add_argument("--dry-run", action="store_true", help="print the job list and one rendered prompt, then exit")
    return p.parse_args()


# ---------------------------------------------------------------- jobs

def seed_for(source, item, lang, cond, sample):
    """Deterministic per-generation seed (crc32 of the key), so reruns of one key are reproducible."""
    return zlib.crc32(f"{source}|{item}|{lang}|{cond}|{sample}".encode())


def make_jobs(prompts, conds, n_samples):
    jobs = []
    for row in prompts.itertuples(index=False):
        for cond in conds:
            for s in range(n_samples[cond]):
                jobs.append(dict(source=row.source, item=int(row.item), lang=row.lang, cond=cond, sample=s,
                                 prompt=row.prompt, seed=seed_for(row.source, row.item, row.lang, cond, s)))
    return jobs


def key(r):
    return (r["source"], int(r["item"]), r["lang"], r["cond"], int(r["sample"]))


def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {key(json.loads(line)) for line in f if line.strip()}


def build_prompt_ids(tok, job):
    """Chat template (thinking on) + optional prefill, tokenized once. Returns (ids, prefill_text)."""
    text = tok.apply_chat_template([{"role": "user", "content": job["prompt"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    # no trailing whitespace after the opener: avoids token-merge issues at the boundary
    prefill = "<think>\n" + TRACE_LANGS[job["lang"]]["opener"] if job["cond"] == "prefill" else ""
    ids = tok(text + prefill, add_special_tokens=False)["input_ids"]
    return ids, prefill


# ---------------------------------------------------------------- record

def distinct_ngram_ratio(ids, n=4):
    """Unique n-grams / total n-grams. ~1 = varied text, low (<0.3) = looping."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return round(len(set(grams)) / len(grams), 2) if grams else None


def make_record(tok, tag, job, prompt_ids, prefill, gen_ids, eos_ids, think_end, finish, secs, backend):
    ids = [t for t in gen_ids if t not in eos_ids]
    if think_end in ids:
        k = ids.index(think_end)
        trace, answer, truncated = ids[:k], ids[k + 1:], False
    else:
        trace, answer, truncated = ids, [], True  # never closed the trace (hit the cap, or stopped oddly)
    return dict(model=tag, source=job["source"], item=job["item"], lang=job["lang"], cond=job["cond"],
                sample=job["sample"], seed=job["seed"], backend=backend,
                prompt_len=len(prompt_ids), prefill=prefill, prompt_ids=prompt_ids, gen_ids=gen_ids,
                n_gen=len(gen_ids), n_trace=len(trace), n_answer=len(answer),
                truncated=truncated, finish=finish, rep_trace=distinct_ngram_ratio(trace),
                rep_answer=distinct_ngram_ratio(answer), secs=round(secs, 2),
                trace_text=tok.decode(trace), answer_text=tok.decode(answer))


def append(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- backends

def run_hf(args, tok, tag, jobs, out):
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    think_end = tok.convert_tokens_to_ids("</think>")
    print(f"hf backend on {device}, eos={eos_ids}, </think>={think_end}")

    for j, job in enumerate(jobs):
        prompt_ids, prefill = build_prompt_ids(tok, job)
        x = torch.tensor([prompt_ids], device=device)
        torch.manual_seed(job["seed"])
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True,
                               max_new_tokens=args.max_new_tokens, **SAMPLING)
        if device == "mps":
            torch.mps.synchronize()
        secs = time.perf_counter() - t0
        gen_ids = y[0, len(prompt_ids):].tolist()
        finish = "length" if len(gen_ids) >= args.max_new_tokens else "stop"
        r = make_record(tok, tag, job, prompt_ids, prefill, gen_ids, eos_ids, think_end, finish, secs, "hf")
        append(out, [r])
        print(f"[{j + 1}/{len(jobs)}] {r['source']} {r['lang']} item={r['item']} {r['cond']} s={r['sample']} "
              f"n_gen={r['n_gen']} trunc={r['truncated']} {secs:.1f}s")


def run_vllm(args, tok, tag, jobs, out):
    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    eos = GenerationConfig.from_pretrained(args.model).eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    think_end = tok.convert_tokens_to_ids("</think>")
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, seed=0)

    for c0 in range(0, len(jobs), args.chunk):
        chunk = jobs[c0:c0 + args.chunk]
        built = [build_prompt_ids(tok, j) for j in chunk]
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _ in built]  # exact ids, no re-tokenizing
        params = [SamplingParams(max_tokens=args.max_new_tokens, seed=j["seed"], **SAMPLING) for j in chunk]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)  # returned in input order
        secs = time.perf_counter() - t0
        recs = []
        for job, (ids, prefill), o in zip(chunk, built, outs):
            c = o.outputs[0]
            recs.append(make_record(tok, tag, job, ids, prefill, list(c.token_ids), eos_ids, think_end,
                                    c.finish_reason, secs / len(chunk), "vllm"))
        append(out, recs)
        n_tok = sum(r["n_gen"] for r in recs)
        print(f"chunk {c0 // args.chunk + 1}: {len(chunk)} gens, {n_tok} tokens, {secs:.0f}s "
              f"({n_tok / secs:.0f} tok/s), truncated {sum(r['truncated'] for r in recs)}", flush=True)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    tag = args.model.split("/")[-1].lower()  # "qwen3-4b"
    out = Path(args.out) if args.out else REPO / f"data/generations/c1_traces_{tag}.jsonl"

    prompts = pd.read_csv(args.prompts)
    if args.langs:
        prompts = prompts[prompts.lang.isin(args.langs.split(","))]
    if args.sources:
        prompts = prompts[prompts.source.isin(args.sources.split(","))]
    if args.limit:
        prompts = prompts.head(args.limit)
    unknown = set(prompts.lang) - set(TRACE_LANGS)
    assert not unknown, f"languages missing from TRACE_LANGS: {unknown}"

    conds = args.conds.split(",")
    jobs = make_jobs(prompts, conds, {"natural": args.n_natural, "prefill": args.n_prefill})
    done = done_keys(out)
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(prompts)} prompts × conds {conds} → {len(jobs)} generations; {len(done)} already in "
          f"{out.name}; {len(todo)} to run")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.dry_run:
        print(pd.DataFrame(todo).groupby(["source", "cond", "lang"]).size().unstack(fill_value=0))
        for cond in conds:
            j = next((j for j in todo if j["cond"] == cond), None)
            if j:
                ids, _ = build_prompt_ids(tok, j)
                print(f"\n--- {cond} | {j['lang']} item {j['item']} | {len(ids)} prompt tokens ---\n"
                      f"{tok.decode(ids)!r}")
        return
    if not todo:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    (run_vllm if args.backend == "vllm" else run_hf)(args, tok, tag, todo, out)
    print(f"done: {len(done_keys(out))} records in {out}")


if __name__ == "__main__":
    main()
