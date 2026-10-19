"""Figures for saved A1 runs (see translation.load_run). Color = role, solid = matched, dashed = control."""

import matplotlib.pyplot as plt
import numpy as np

from latent_switch.stats import mean_ci, median_iqr, paired_diff, rank_advantage


def _band(ax, layers, stat, color, ls="-", label=None):
    mid, lo, hi = stat
    ax.plot(layers, mid, color=color, ls=ls, label=label)
    ax.fill_between(layers, lo, hi, color=color, alpha=0.15)


def _finish(fig, axes, run):
    cfg = run["config"]
    for a in axes:
        a.set_xlabel("layer")
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.suptitle(f"{cfg['model']} · {cfg['src']} → {cfg['tgt']}")
    fig.tight_layout()


def plot_run(run: dict, measure: str = "first"):
    """(a) P(matched/control words), correct-only · (b) ranks · (c) script mass · (d) entropy."""
    c, roles = run["curves"], run["roles"]
    ok = np.array(run["correct"])
    layers = np.arange(c["entropy"].shape[1])
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))

    for k, (r, lang) in enumerate(roles.items()):
        for suffix, ls, kind in [("", "-", "matched"), ("_ctrl", "--", "control")]:
            label = f"{lang} {kind}"
            _band(ax[0], layers, mean_ci(c[f"{r}{suffix}_{measure}"].numpy()[ok]), f"C{k}", ls, label)
            _band(ax[1], layers, median_iqr(c[f"rank_{r}{suffix}"].numpy()[ok] + 1), f"C{k}", ls, label)
    ax[0].set(title=f"(a) P({measure} token), correct only (n={ok.sum()})", ylabel="probability")
    ax[1].set(title="(b) rank + 1 of first token (median, IQR), correct only", yscale="log", ylabel="rank + 1")
    ax[1].invert_yaxis()                                  # better rank = higher

    for k, s in enumerate(run["scripts"]):
        _band(ax[2], layers, mean_ci(c[f"script_{s}"].numpy()), f"C{k + len(roles)}", label=s)
    ax[2].set(title=f"(c) P(any token of script), all (n={len(ok)})", ylabel="probability")

    _band(ax[3], layers, mean_ci(c["entropy"].numpy()), "k")
    ax[3].set(title="(d) entropy (bits), all", ylabel="bits")
    _finish(fig, ax, run)
    return fig


def plot_paired(run: dict, measure: str = "first"):
    """Paired matched − control per prompt, mean ± 95% CI, correct-only. Above 0 = concept-specific signal."""
    c, roles = run["curves"], run["roles"]
    ok = np.array(run["correct"])
    layers = np.arange(c["entropy"].shape[1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, (r, lang) in enumerate(roles.items()):
        _band(ax[0], layers, mean_ci(paired_diff(c, r, measure)[ok]), f"C{k}", label=lang)
        _band(ax[1], layers, mean_ci(rank_advantage(c, r)[ok]), f"C{k}", label=lang)
    for a in ax:
        a.axhline(0, color="k", lw=0.8)
    ax[0].set(title=f"(a) P matched − P control ({measure}), n={ok.sum()}", ylabel="Δ probability")
    ax[1].set(title="(b) rank advantage: log10(rank ctrl+1) − log10(rank+1)", ylabel="orders of magnitude")
    _finish(fig, ax, run)
    return fig
