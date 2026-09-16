from __future__ import annotations

import sys
import collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LABEL = {"gpt-oss-120b": "gpt-oss-120b", "qwen3-nothink": "Qwen3-32B",
         "deepseek-v4-flash": "DeepSeek-V4-Flash",
         "deepseek-v4-pro-think": "DeepSeek-V4-Pro (r)",
         "deepseek-v4-pro": "DeepSeek-V4-Pro", "kimi-k3": "Kimi-K3",
         "minimax-m3-think": "MiniMax-M3 (r)", "minimax-m3-nothink": "MiniMax-M3",
         "glm-5.2-think": "GLM-5.2 (r)", "glm-5.2-nothink": "GLM-5.2",
         "llama-3.3-70b": "Llama-3.3-70B",
         "gpt-5.6-think": "GPT-5.6 (r)", "gpt-5.6-nothink": "GPT-5.6",
         "sonnet-5-think": "Sonnet-5 (r)", "sonnet-5-nothink": "Sonnet-5"}
ORDER = ["gpt-oss-120b", "deepseek-v4-flash", "deepseek-v4-pro-think", "kimi-k3",
         "gpt-5.6-think", "deepseek-v4-pro", "minimax-m3-think", "sonnet-5-nothink",
         "sonnet-5-think", "qwen3-nothink", "gpt-5.6-nothink",
         "minimax-m3-nothink", "glm-5.2-nothink", "glm-5.2-think", "llama-3.3-70b"]
HUMAN = 0.516
GREY = ["#1a1a1a", "#5a5a5a", "#9a9a9a", "#c8c8c8"]

BLUE, VERM, GREEN, ORANGE, PURPLE = ("#0072B2", "#D55E00", "#009E73",
                                     "#E69F00", "#CC79A7")
COLOR = "--bw" not in sys.argv

def _place_labels(ax, pts, colors, dx=0.018, min_gap=0.045, side=None):
    order = sorted(range(len(pts)), key=lambda i: -pts[i][2])
    taken = []
    for i in order:
        name, x, y = pts[i][0], pts[i][1], pts[i][2]
        ly = y
        while any(abs(ly - t) < min_gap for t in taken):
            ly -= min_gap * 0.55
        taken.append(ly)
        right = (x < 0.55) if side is None else (side == 'right')
        ax.annotate(name, (x, y), (x + (dx if right else -dx), ly),
                    fontsize=6.5, color=colors[i], va="center",
                    ha="left" if right else "right", annotation_clip=False,
                    arrowprops=(dict(arrowstyle="-", lw=0.5, color=colors[i],
                                     alpha=0.45, shrinkA=0, shrinkB=3)
                                if abs(ly - y) > 0.012 else None))

TREAT_COLOR = {"no_history": "#9a9a9a", "social_history_neg": "#D55E00",
               "communication": "#0072B2", "social_history_pos": "#009E73"}

def fig_A(data):
    conds = [("no_history", "Baseline"), ("social_history_neg", "Trust exploited"),
             ("communication", "Communication"), ("social_history_pos", "Trust rewarded")]
    models = [m for m in ORDER if m in data]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y = np.arange(len(models))
    h = 0.2
    for k, (c, lab) in enumerate(conds):
        v = [np.mean([A.frac(r) for r in A.sel(data[m], None, c)]) for m in models]
        ax.barh(y + (k - 1.5) * h, v, height=h, label=lab,
                color=(TREAT_COLOR[c] if COLOR else GREY[k]),
                edgecolor="white", linewidth=0.3)
    ax.axvline(HUMAN, color="black", ls=(0, (4, 2)), lw=0.9)
    ax.annotate("human benchmark", xy=(HUMAN, -0.85), xytext=(HUMAN + 0.02, -0.85),
                fontsize=6.5, va="center", annotation_clip=False)
    ax.set_yticks(y, [LABEL[m] for m in models])
    ax.set_xlabel("mean transfer, fraction of the endowment")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(models) - 0.5, -1.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=4,
              handlelength=1.2, columnspacing=1.4)
    fig.savefig(OUT / "fig_A_treatment_landscape.pdf")
    plt.close(fig)
    return "fig_A_treatment_landscape.pdf"

def fig_B(data):
    models = [m for m in ORDER if m in data]
    pairs = [("canonical", "reskin", "labels only"),
             ("shifted", "reskin_shifted", "labels only, shifted payoffs"),
             ("strong_control", "strong_novel", "whole narrative replaced")]
    fig, ax = plt.subplots(figsize=(3.6, 3.7))
    marks = ["o", "s", "^"]
    cols = [BLUE, GREEN, VERM] if COLOR else GREY[:3]
    for k, (a, b, lab) in enumerate(pairs):
        d, e = [], []
        for m in models:
            ga = [A.frac(r) for r in A.sel(data[m], a, "no_history")]
            gb = [A.frac(r) for r in A.sel(data[m], b, "no_history")]
            c = A.contrast(ga, gb)
            d.append(c["delta"])
            e.append((c["ci"][1] - c["ci"][0]) / 2)
        yy = np.arange(len(models)) + (k - 1) * 0.24
        ax.errorbar(d, yy, xerr=e, fmt=marks[k], ms=3.6, lw=0, elinewidth=0.8,
                    capsize=1.5, color=cols[k], mec=cols[k], label=lab)
    ax.axvline(0, color="black", lw=0.8)
    for i in range(len(models)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000", alpha=0.035, lw=0,
                       zorder=0)
    ax.set_yticks(np.arange(len(models)), [LABEL[m] for m in models])
    ax.set_xlabel("recognizable $-$ de-recognized wording,\nfraction of the endowment")
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), frameon=False,
              ncol=1, handletextpad=0.4)
    fig.savefig(OUT / "fig_B_surface_gradient.pdf")
    plt.close(fig)
    return "fig_B_surface_gradient.pdf"

def fig_C():
    fams = ["DeepSeek-V4-Pro", "MiniMax-M3", "GPT-5.6", "Sonnet-5", "GLM-5.2"]
    short = ["DeepSeek", "MiniMax", "GPT-5.6", "Sonnet-5", "GLM-5.2"]
    text = [0.841, 0.733, 0.704, 0.626, 0.687]
    dec = [0.640, 0.722, 0.704, 0.638, 0.596]
    plac = [0.512, 0.506, 0.470, 0.474, 0.477]
    zero_r = [0.538, 0.361, 0.359, 0.145, 0.219]
    zero_p = [0.319, 0.103, 0.146, 0.005, 0.211]

    x = np.arange(len(fams))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    c_text, c_dec = (BLUE, GREEN) if COLOR else (GREY[0], GREY[2])
    ax.bar(x - 0.19, text, 0.36, label="stated reasoning", color=c_text,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.19, dec, 0.36, label="the two decisions", color=c_dec,
           edgecolor="white", linewidth=0.4)
    for i, p in enumerate(plac):
        ax.plot([i - 0.42, i + 0.42], [p, p], color="black", lw=0.9,
                ls=(0, (3, 2)))

    ax.set_xticks(x, short)
    ax.set_ylabel("AUC, telling the two arms apart")
    ax.set_ylim(0.45, 0.92)
    ax.set_xlim(-0.55, len(fams) - 0.45)
    ax.tick_params(axis="x", labelrotation=18)
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color="black", lw=0.9, ls=(0, (3, 2))))
    l.append("placebo")
    ax.legend(h, l, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              ncol=3, handlelength=1.3, columnspacing=1.0, handletextpad=0.5)

    ax = axes[1]
    c_on, c_off = (VERM, "#9a9a9a") if COLOR else (GREY[0], GREY[2])
    ax.bar(x - 0.19, zero_p, 0.36, label="reasoning off", color=c_off,
           edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.19, zero_r, 0.36, label="reasoning on", color=c_on,
           edgecolor="white", linewidth=0.4)
    for i in range(len(fams)):
        if zero_p[i] > 0 and zero_r[i] / zero_p[i] > 1.5:
            ax.annotate("", xy=(i + 0.19, zero_r[i] + 0.012),
                        xytext=(i - 0.19, zero_p[i] + 0.012),
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color=c_on, shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=-0.35"))
    ax.set_xticks(x, short)
    ax.set_ylabel("share of dyads transferring nothing")
    ax.set_ylim(0, 0.63)
    ax.set_xlim(-0.55, len(fams) - 0.45)
    ax.tick_params(axis="x", labelrotation=18)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
              ncol=2, handlelength=1.1, columnspacing=1.0)

    fig.subplots_adjust(wspace=0.34)
    fig.savefig(OUT / "fig_C_toggle_anatomy.pdf")
    plt.close(fig)
    return "fig_C_toggle_anatomy.pdf"

def fig_D(data):
    import json
    models = [m for m in ORDER if m in data]
    obs, bnd = [], []
    for m in models:
        p_ = [A.frac(r) for r in A.sel(data[m], None, "social_history_pos")]
        n_ = [A.frac(r) for r in A.sel(data[m], None, "social_history_neg")]
        lost = collections.Counter()
        arch = A.DATA / "full" / f"full_{m}_purged.jsonl"
        if arch.exists():
            for line in arch.read_text().splitlines():
                if line.strip():
                    lost[json.loads(line)["condition"]] += 1
        obs.append(np.mean(p_) - np.mean(n_))
        pb = p_ + [0.0] * lost["social_history_pos"]
        nb = n_ + [1.0] * lost["social_history_neg"]
        bnd.append(np.mean(pb) - np.mean(nb))
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    obs_c, bnd_c = (BLUE, VERM) if COLOR else (GREY[0], GREY[0])
    for i in y:
        fails = bnd[i] < 0.05
        ax.plot([bnd[i], obs[i]], [i, i],
                color=(VERM if (fails and COLOR) else GREY[2]),
                lw=(1.8 if fails else 1.2), zorder=1, alpha=0.9)
    ax.scatter(bnd, y, s=42, color="white", edgecolor=bnd_c, linewidth=1.2,
               zorder=3, label="worst case bound")
    ax.scatter(obs, y, s=16, color=obs_c, zorder=4, label="observed")
    ax.axvline(0, color="black", lw=0.8)
    for i in range(len(models)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000", alpha=0.035, lw=0,
                       zorder=0)
    ax.set_yticks(y, [LABEL[m] for m in models])
    ax.set_xlabel("favorable $-$ unfavorable report,\nfraction of the endowment")
    ax.set_ylim(len(models) - 0.5, -1.3)
    ax.set_xlim(-0.04, 0.82)
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[1], h[0]], [l[1], l[0]], frameon=False, loc="lower left",
              bbox_to_anchor=(-0.02, 1.01), ncol=2, handlelength=1.0,
              columnspacing=1.4, handletextpad=0.4)
    fig.savefig(OUT / "fig_D_attrition_bound.pdf")
    plt.close(fig)
    return "fig_D_attrition_bound.pdf"

def fig_E(data):
    models = [m for m in ORDER if m in data]
    rows = []
    for m in models:
        f = np.array([A.frac(r) for r in data[m]])
        rows.append((m, (f == 0).mean(), ((f > 0) & (f < 0.999)).mean(),
                     (f >= 0.999).mean(), f.mean()))
    rows.sort(key=lambda r: -r[1])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    z = np.array([r[1] for r in rows])
    mid = np.array([r[2] for r in rows])
    top = np.array([r[3] for r in rows])
    cz, cm, ct = (VERM, "#c8c8c8", GREEN) if COLOR else (GREY[0], GREY[2], GREY[1])
    ax.barh(y, z, color=cz, edgecolor="white", linewidth=0.4, label="transfers nothing")
    ax.barh(y, mid, left=z, color=cm, edgecolor="white", linewidth=0.4,
            label="transfers part")
    ax.barh(y, top, left=z + mid, color=ct, edgecolor="white", linewidth=0.4,
            label="transfers everything")
    ax.scatter([r[4] for r in rows], y, s=26, color=BLUE if COLOR else "black",
               zorder=4, marker="D", label="mean transfer")
    ax.set_yticks(y, [LABEL[r[0]] for r in rows])
    ax.set_xlabel("share of dyads (bars) and the mean transfer (diamond)")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.5, -1.6)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), frameon=False,
              ncol=2, handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
    fig.savefig(OUT / "fig_E_distributions.pdf")
    plt.close(fig)
    return "fig_E_distributions.pdf"

def fig_F():
    import json, collections
    codes = collections.defaultdict(list)
    for line in (Path(__file__).resolve().parent / "data" / "coding"
                 / "judge_comm.jsonl").read_text().splitlines():
        j = json.loads(line)
        if j.get("coder") == "judge:judge-gptoss" and j.get("coded_clean"):
            codes[j["model_key"]].append(j)
    pts = []
    for m, rows in codes.items():
        if m not in LABEL:
            continue
        ag = [r for r in rows if r.get("explicit_agreement")]
        amt = [r for r in ag if r.get("agreed_send") is not None]
        if len(amt) < 30:
            continue
        kept = sum(1 for r in amt if r.get("sent") == r.get("agreed_send"))
        pts.append((LABEL[m], len(ag) / len(rows), kept / len(amt), len(amt)))
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    ax.axhline(0.5, color="#cccccc", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.02, 0.52, "keeps more often than not", fontsize=6, color="#999999")
    cols = [((VERM if (yv < 0.5 and x > 0.5) else BLUE) if COLOR else GREY[0])
            for _, x, yv, _ in pts]
    for (name, x, yv, n), c in zip(pts, cols):
        ax.scatter([x], [yv], s=34, color=c, zorder=3)
    _place_labels(ax, [(n_, x, y) for n_, x, y, _ in pts], cols,
                  dx=0.022, min_gap=0.052, side='right')
    ax.set_xlabel("share of transcripts reaching an explicit agreement")
    ax.set_ylabel("share of agreements the first mover kept")
    ax.set_xlim(-0.02, 1.62)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.spines["bottom"].set_bounds(0, 1.0)
    missing = [LABEL[m] for m in ORDER
               if m in LABEL and m not in codes and m not in ("qwen3-think",)]
    if missing:
        ax.text(0.0, -0.175, "awaiting coding: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=6, color="#777777")
    fig.savefig(OUT / "fig_F_agree_vs_comply.pdf")
    plt.close(fig)
    return "fig_F_agree_vs_comply.pdf"

def fig_G():
    v = [("agreement present", 0.783, 0.578), ("agreement type", 0.805, 0.598),
         ("equality framing", 0.959, 0.681), ("promise, first mover", 0.929, 0.315),
         ("promise, second mover", 0.875, 0.559)]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    lo, hi = 0.74, 1.00
    ax.plot([lo, hi], [lo, hi], color="#bbbbbb", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.915, 0.933, "equal agreement", fontsize=6, color="#999999",
            rotation=38, rotation_mode="anchor")
    ax.axhline(0.70, color="#999999", lw=0.7, ls=(0, (1, 2)), zorder=1)
    ax.text(lo + 0.005, 0.715, "conventional threshold", fontsize=6, color="#777777")
    WITHDRAWN = {"promise, first mover", "promise, second mover"}
    cols = [((VERM if name in WITHDRAWN else BLUE) if COLOR else GREY[0])
            for name, _, _ in v]
    for (name, mm, mh), c in zip(v, cols):
        ax.plot([mm, mm], [mh, mm], color=c, lw=0.9, alpha=0.4, zorder=2)
        ax.scatter([mm], [mh], s=36, color=c, zorder=3)
    _place_labels(ax, [(n_, x, y) for n_, x, y in v], cols,
                  dx=0.012, min_gap=0.055, side="right")
    ax.set_xlabel("agreement between two runs of the judge")
    ax.set_ylabel("agreement with a human reading")
    ax.set_xlim(0.74, 1.00)
    ax.set_ylim(0.25, 1.00)
    fig.savefig(OUT / "fig_G_reliability_validity.pdf")
    plt.close(fig)
    return "fig_G_reliability_validity.pdf"

def main() -> None:
    want = [a.upper() for a in sys.argv[1:] if not a.startswith("--")] \
        or ["A", "B", "C", "D", "E", "F", "G"]
    data = A.load()
    made = []
    if "A" in want: made.append(fig_A(data))
    if "B" in want: made.append(fig_B(data))
    if "C" in want: made.append(fig_C())
    if "D" in want: made.append(fig_D(data))
    if "E" in want: made.append(fig_E(data))
    if "F" in want: made.append(fig_F())
    if "G" in want: made.append(fig_G())
    for f in made:
        print(f"  {OUT / f}")
    print("\nEach figure carries one claim. Pick the claim the paper should make\n"
          "visually, since the supervisor asked for a figure and not for four.")

if __name__ == "__main__":
    main()
