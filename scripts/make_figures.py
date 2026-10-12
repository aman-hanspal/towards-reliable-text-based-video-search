"""
make_figures.py  --  build every paper figure (and the main table) from the RAW result files.   (CPU only)

Every number is recomputed here from the files the experiments wrote; nothing is typed in by hand.
At the end it prints a CHECK table: each recomputed number next to the value in the first draft. If a line says
'DIFF', tell Claude before using the figure.

Reads (from the OVT-B-Dataset folder; it only READS these):
  scores_full/prompt_scores.csv                       detector scores, all 9 runs        -> Figs 2, 3, 4, table
  full_runs/<run>/timing.csv                          speed                              -> Fig 4, table
  frame_rate_cuts/<run>_rows.csv                      fewer-frames scores (3 sizes)      -> Fig 5
  vlm_sets/<run>/items.csv (+ crops/)                 VLM test sets                      -> Figs 6, 8
  vlm_runs/.../answers.csv                            VLM answers (6 settings)           -> Fig 6
  vlm_results/hand_check/B_check.csv                  Aman's reasons for rejections      -> Fig 7, 8

Writes only into paper_figures/:
  fig1_pipeline  fig2_main  fig3_joint  fig4_speed  fig5_frames  fig6_vlm  fig7_reasons  fig8_examples
  (each as .png for viewing and .pdf for the paper)
  table1_main.csv / .md      the main table, incl. the 480 px easy / strict-hard numbers missing from the draft
  numbers.csv                every number behind every figure
  check.md                   recomputed vs draft values

Usage:
  python make_figures.py
  python make_figures.py --fig8 A_1234_5,A_99_6,B_1802_0,B_77_1     # choose the four example crops yourself
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

THR = 0.25
SIZES = ("small", "medium", "large")
LETTER = {"small": "s", "medium": "m", "large": "l"}
SEP = {(s, r): f"yolov8{LETTER[s]}-worldv2_img{r}_fp16_separate" for s in SIZES for r in (640, 480)}
JOINT = {s: f"yolov8{LETTER[s]}-worldv2_img640_fp16_joint" for s in SIZES}
MAIN = SEP[("medium", 640)]
TAG = "rtx3050__qwen3-vl-2b-instruct"
OUT = Path("paper_figures")

# values in the first draft (11 Oct), only used for the CHECK table
DRAFT = {
    ("found", "small", 640): 53.3, ("found", "medium", 640): 50.7, ("found", "large", 640): 52.2,
    ("found", "small", 480): 52.0, ("found", "medium", 480): 50.5, ("found", "large", 480): 51.9,
    ("easy", "small", 640): 2.5, ("easy", "medium", 640): 1.1, ("easy", "large", 640): 1.1,
    ("hard", "small", 640): 28.9, ("hard", "medium", 640): 23.9, ("hard", "large", 640): 24.6,
    ("hard", "small", 480): 27.1, ("hard", "medium", 480): 24.0, ("hard", "large", 480): 23.2,
    ("strict", "small", 640): 33.8, ("strict", "medium", 640): 31.0, ("strict", "large", 640): 31.6,
    ("frame_hit", "medium", 640): 38.5, ("misnaming_share", "medium", 640): 94.0,
    ("ms", "small", 640): 25.1, ("ms", "medium", 640): 34.1, ("ms", "large", 640): 48.2,
    ("ms", "small", 480): 25.3, ("ms", "medium", 480): 32.4, ("ms", "large", 480): 40.6,
    ("joint_diff", "small", 640): -9.9, ("joint_diff", "medium", 640): -8.4, ("joint_diff", "large", 640): -8.3,
    ("vlm_after", "small", 640): 9.1, ("vlm_after", "medium", 640): 8.1, ("vlm_after", "large", 640): 9.0,
    ("vlm_after", "small", 480): 8.7, ("vlm_after", "medium", 480): 8.3, ("vlm_after", "large", 480): 8.3,
    ("thr_matched", "small", 640): 20.7, ("thr_matched", "medium", 640): 18.7, ("thr_matched", "large", 640): 16.5,
    ("thr_matched", "small", 480): 19.4, ("thr_matched", "medium", 480): 17.4, ("thr_matched", "large", 480): 14.8,
    ("vlm_removed_A", "medium", 640): 69.0, ("vlm_lost_B", "medium", 640): 10.2,
    ("found_k10", "small", 640): 45.8, ("found_k10", "medium", 640): 44.5, ("found_k10", "large", 640): 46.3,
    ("hard_k10", "small", 640): 21.0, ("hard_k10", "medium", 640): 17.8, ("hard_k10", "large", 640): 18.6,
}

# reference palette, light mode (first three slots validated for colour-blind separation) + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#b4b2ab", "#0b0b0b", "#52514e", "#e6e5e1"
SIZE_COL = {"small": BLUE, "medium": ORANGE, "large": AQUA}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})

NUMBERS = {}      # (measure, size, res) -> value, written to numbers.csv


def note(key, value):
    NUMBERS[key] = value
    return value


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {OUT / name}.png / .pdf")


# ===================================================================== loading
def load_scores(path):
    """{run: {(video_id, prompt_idx): row}} at THR, only the fields needed."""
    keep = set(SEP.values()) | set(JOINT.values())
    out = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["run"] not in keep or abs(float(r["threshold"]) - THR) > 1e-9:
                continue
            out[r["run"]][(int(r["video_id"]), int(r["prompt_idx"]))] = {
                "kind": r["kind"], "strict": r["strict_hard"] == "True",
                "gt": r["gt_boxes"] not in ("", "0"),
                "found": int(r["video_found"]) if r["video_found"] not in ("", None) else 0,
                "hit": float(r["frame_hit_rate"]) if r["frame_hit_rate"] not in ("", None) else None,
                "fired": int(r["fired"]) if r["fired"] not in ("", None) else 0,
                "mis": int(r["fired_misnaming"]) if r["fired_misnaming"] not in ("", None) else 0}
    return out


def sel(rows, what):
    if what == "present":
        return {k: v for k, v in rows.items() if v["kind"] == "present" and v["gt"]}
    if what == "easy":
        return {k: v for k, v in rows.items() if v["kind"] == "easy_absent"}
    if what == "hard":
        return {k: v for k, v in rows.items() if v["kind"] == "hard_absent"}
    if what == "strict":
        return {k: v for k, v in rows.items() if v["kind"] == "hard_absent" and v["strict"]}


def mean_of(rows, field):
    xs = [v[field] for v in rows.values() if v[field] is not None]
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def paired_diff(a, b, field, boot=2000, seed=20261011):
    """(mean_b - mean_a) in points with a 95% bootstrap interval over videos; a, b share keys."""
    keys = sorted(set(a) & set(b))
    vids = sorted({k[0] for k in keys})
    vi = {v: i for i, v in enumerate(vids)}
    sa, sb, c = np.zeros(len(vids)), np.zeros(len(vids)), np.zeros(len(vids))
    for k in keys:
        i = vi[k[0]]
        sa[i] += a[k][field]; sb[i] += b[k][field]; c[i] += 1
    d0 = 100 * (sb.sum() - sa.sum()) / c.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(boot, len(vids)))
    d = 100 * (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return d0, np.percentile(d, 2.5), np.percentile(d, 97.5)


def speed_ms(run):
    p = Path("full_runs") / run / "timing.csv"
    if not p.exists():
        return None
    wall = fp = 0.0
    for r in csv.DictReader(open(p, encoding="utf-8")):
        wall += float(r["wall_s"]); fp += float(r["frames"]) * float(r["passes"])
    return 1000 * wall / fp if fp else None


def vlm_numbers(run, scores):
    """Same rules as score_vlm.py: pilot excluded, anything but 'yes' rejects; after = before x share kept."""
    set_dir = Path("vlm_sets") / run
    cands = [Path("vlm_runs") / TAG / "answers.csv"] if run == MAIN else []
    cands.append(Path("vlm_runs/by_run") / run / TAG / "answers.csv")
    ans_path = next((p for p in cands if p.exists()), None)
    if not (set_dir / "items.csv").exists() or ans_path is None:
        print(f"  (VLM files for {run} not found: skipped)")
        return None
    items = {r["item_id"]: r for r in csv.DictReader(open(set_dir / "items.csv", encoding="utf-8"))}
    ans = {r["item_id"]: r["answer"] for r in csv.DictReader(open(ans_path, encoding="utf-8"))}
    rows = [(items[i], a) for i, a in ans.items() if i in items and not items[i]["set"].startswith("pilot_")]
    keep = lambda f: (lambda s: sum(1 for _, a in s if a == "yes") / len(s) if s else None)([(it, a) for it, a in rows if f(it)])
    kb = keep(lambda it: it["set"] == "B")
    kA = keep(lambda it: it["set"] == "A")
    rs = scores[run]
    r_found = mean_of(sel(rs, "present"), "found") / 100
    r_hard = mean_of(sel(rs, "hard"), "fired") / 100
    pool_a = sum(1 for i in items.values() if i["set"] in ("A", "pilot_A") and i["kind"] == "hard_absent")
    pool_c = sum(1 for i in items.values() if i["set"] == "C" and i["kind"] == "hard_absent")
    ka = keep(lambda it: it["set"] == "A" and it["kind"] == "hard_absent")
    kc = keep(lambda it: it["set"] == "C" and it["kind"] == "hard_absent")
    kept = (pool_a * (ka if ka is not None else 1) + pool_c * (kc if kc is not None else 1)) / (pool_a + pool_c)
    # stricter detector threshold that loses the same share of set B
    b_scores = sorted(float(it["yolo_score"]) for it, _ in rows if it["set"] == "B")
    t2 = THR
    for s in [THR] + b_scores:
        if sum(1 for x in b_scores if x >= s) / len(b_scores) <= kb + 1e-12:
            t2 = s
            break
    else:
        t2 = max(b_scores) + 1e-6
    hard_items = [i for i in items.values() if i["kind"] == "hard_absent" and i["set"] in ("A", "pilot_A", "C")]
    score_of = lambda i: float(i.get("prompt_max_score") or i["yolo_score"])
    still = sum(1 for i in hard_items if score_of(i) >= t2)
    print(f"  VLM {run}: answers from {ans_path}")
    return {"before": 100 * r_hard, "after": 100 * r_hard * kept, "thr": t2,
            "thr_rate": 100 * r_hard * still / max(1, len(hard_items)),
            "removed_A": 100 * (1 - kA), "lost_B": 100 * (1 - kb), "found_after": 100 * r_found * kb}


def frame_cut(run):
    p = Path("frame_rate_cuts") / f"{run}_rows.csv"
    if not p.exists():
        return None
    base, found, hard = set(), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if abs(float(r["threshold"]) - THR) < 1e-9]
    for r in rows:
        if r["k"] == "1" and r["kind"] == "present" and r["gt_boxes"] not in ("", "0"):
            base.add((r["video_id"], r["prompt_idx"]))
    for r in rows:
        k = int(r["k"])
        if r["kind"] == "present" and (r["video_id"], r["prompt_idx"]) in base:
            found[k][0] += int(r["video_found"] or 0); found[k][1] += 1
        elif r["kind"] == "hard_absent":
            hard[k][0] += int(r["fired"] or 0); hard[k][1] += 1
    ks = sorted(found)
    return ks, [100 * found[k][0] / found[k][1] for k in ks], [100 * hard[k][0] / hard[k][1] for k in ks]


# ===================================================================== figures
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 2.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    boxes = [("Video\n(OVT-B,\n1,972 videos)", "#eef3fb"), ("YOLO-World v2\none prompt\nat a time", "#eef3fb"),
             ("Score vs labels:\npresent / unrelated\n/ lookalike", "#eef3fb"),
             ("Crop top box,\nask Qwen3-VL 2B:\n'main object a X?'", "#fdf0ea"), ("Keep only\n'yes'", "#eaf7f1")]
    w, gap = 16.4, 4.0
    for i, (t, c) in enumerate(boxes):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 6), w, 18, boxstyle="round,pad=0.3,rounding_size=1.5", fc=c, ec=INK2, lw=0.8))
        ax.text(x + w / 2, 15, t, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.3)
        if i:
            ax.annotate("", xy=(x - 0.6, 15), xytext=(x - gap + 0.6, 15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    ax.text(31, 1.2, "evaluation (Sections 3.1-3.4)", ha="center", fontsize=7, color=INK2)
    ax.text(81, 1.2, "proposed check (Section 3.5)", ha="center", fontsize=7, color=INK2)
    save(fig, "fig1_pipeline")


def fig_main(scores):
    measures = [("found", "found in video"), ("easy", "unrelated\nfalse alarms"), ("hard", "lookalike\nfalse alarms"),
                ("strict", "strict lookalike\nfalse alarms")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w = 0.26
    for i, s in enumerate(SIZES):
        rs = scores[SEP[(s, 640)]]
        vals = [note((m, s, 640), mean_of(sel(rs, "present" if m == "found" else m), "found" if m == "found" else "fired"))
                for m, _ in measures]
        xs = [j + (i - 1) * w for j in range(4)]
        ax.bar(xs, vals, w, color=SIZE_COL[s], edgecolor="white", linewidth=1.5, label=s)
        for x, y in zip(xs, vals):
            ax.text(x, y + 0.8, f"{y:.1f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([m[1] for m in measures])
    ax.set_ylabel(f"% of prompts (threshold {THR})"); ax.set_ylim(0, 62); ax.grid(axis="x", visible=False)
    ax.legend(title="YOLO-World size (640 px)", ncol=3, loc="upper right", fontsize=7.5, title_fontsize=7.5)
    save(fig, "fig2_main")


def fig_joint(scores):
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    w = 0.34
    for i, s in enumerate(SIZES):
        if JOINT[s] not in scores:
            continue
        a, b = sel(scores[SEP[(s, 640)]], "hard"), sel(scores[JOINT[s]], "hard")
        sep_rate = mean_of(a, "fired")
        d, lo, hi = paired_diff(a, b, "fired")
        note(("joint_diff", s, 640), d); note(("joint_diff_lo", s, 640), lo); note(("joint_diff_hi", s, 640), hi)
        ax.bar(i - w / 2, sep_rate, w, color=BLUE, edgecolor="white", linewidth=1.5, label="one prompt at a time" if i == 0 else None)
        ax.bar(i + w / 2, sep_rate + d, w, color=ORANGE, edgecolor="white", linewidth=1.5, label="all prompts at once" if i == 0 else None)
        ax.text(i + w / 2, sep_rate + d + 0.8, f"{d:+.1f}\n[{lo:+.1f}, {hi:+.1f}]", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(SIZES); ax.set_ylim(0, 37)
    ax.set_ylabel("lookalike false alarms (%)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, "fig3_joint")


def fig_speed(scores):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for (s, r), run in SEP.items():
        ms = speed_ms(run)
        if ms is None or run not in scores:
            continue
        note(("ms", s, r), ms)
        hard = note(("hard", s, r), mean_of(sel(scores[run], "hard"), "fired"))
        found = note(("found", s, r), mean_of(sel(scores[run], "present"), "found"))
        for ax, y in ((axs[0], hard), (axs[1], found)):
            ax.scatter(ms, y, s=46, color=SIZE_COL[s], marker="o" if r == 640 else "s", edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(f"{s[0].upper()}{r}", (ms, y), xytext=(-26, 4) if (s, r) == ("medium", 480) else (4, 3),
                        textcoords="offset points", fontsize=6.5, color=INK2)
    axs[0].set_ylabel("lookalike false alarms (%)"); axs[1].set_ylabel("found in video (%)")
    for ax in axs:
        ax.set_xlabel("ms per frame per prompt (RTX 3050 Laptop, FP16)")
    h = [plt.Line2D([], [], marker="o", ls="", color=SIZE_COL[k], label=k) for k in SIZES] + \
        [plt.Line2D([], [], marker="o", ls="", color=GRAY, label="640 px"), plt.Line2D([], [], marker="s", ls="", color=GRAY, label="480 px")]
    fig.legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig4_speed")


def fig_frames():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    got = False
    for s in SIZES:
        res = frame_cut(SEP[(s, 640)])
        if res is None:
            continue
        got = True
        ks, fnd, hrd = res
        note(("found_k10", s, 640), fnd[-1]); note(("hard_k10", s, 640), hrd[-1])
        for ax, v in ((axs[0], fnd), (axs[1], hrd)):
            ax.plot(range(len(ks)), v, color=SIZE_COL[s], lw=2, marker="o", ms=5, mec="white", mew=1, label=s)
            ax.text(len(ks) - 1 + 0.08, v[-1], s, fontsize=6.8, color=INK2, va="center")
    if not got:
        print("  (frame_rate_cuts/*_rows.csv not found: Fig 5 skipped)")
        plt.close(fig)
        return
    for ax, lab in ((axs[0], "found in video (%)"), (axs[1], "lookalike false alarms (%)")):
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"every {x}\n({100 // x}%)" if x > 1 else "every\nframe" for x in ks])
        ax.set_ylabel(lab); ax.set_xlim(-0.2, len(ks) - 0.4)
    axs[0].legend(loc="lower left", fontsize=7)
    save(fig, "fig5_frames")


def fig_vlm(scores):
    order = [("small", 640), ("medium", 640), ("large", 640), ("small", 480), ("medium", 480), ("large", 480)]
    res = {k: vlm_numbers(SEP[k], scores) for k in order if SEP[k] in scores}
    res = {k: v for k, v in res.items() if v}
    if not res:
        print("  (no VLM results found: Fig 6 skipped)")
        return
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = 0.26
    for i, k in enumerate(order):
        if k not in res:
            continue
        v = res[k]
        for key, val in (("vlm_before", v["before"]), ("vlm_after", v["after"]), ("thr_matched", v["thr_rate"]),
                         ("thr_value", v["thr"]), ("vlm_removed_A", v["removed_A"]), ("vlm_lost_B", v["lost_B"]),
                         ("found_after_vlm", v["found_after"])):
            note((key, k[0], k[1]), val)
        for j, (val, col, lab) in enumerate(((v["before"], GRAY, "no check"),
                                             (v["thr_rate"], ORANGE, "stricter YOLO threshold (same loss)"),
                                             (v["after"], BLUE, "VLM check"))):
            ax.bar(i + (j - 1) * w, val, w, color=col, edgecolor="white", linewidth=1.5, label=lab if i == 0 else None)
        ax.text(i + w, v["after"] + 0.6, f"{v['after']:.1f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{s[0].upper()}{r}" for s, r in order])
    ax.set_ylabel("lookalike false alarms (%)"); ax.set_ylim(0, 35); ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right", fontsize=7)
    save(fig, "fig6_vlm")


REASON_LABEL = {"hand": "only a hand of the 'person' visible", "surface": "object covered by something on it",
                "competing": "another held object equally prominent", "miss": "VLM simply wrong",
                "other": "dataset label wrong"}


def fig_reasons():
    p = Path("vlm_results/hand_check/B_check.csv")
    if not p.exists():
        print("  (B_check.csv not found: Fig 7 skipped)")
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["reason"].strip().lower() or "(blank)"] += 1
    order = [k for k in REASON_LABEL if k in cnt] + [k for k in cnt if k not in REASON_LABEL]
    for k in order:
        note(("reason_" + k, "medium", 640), cnt[k])
    fig, ax = plt.subplots(figsize=(5.6, 2.2))
    ys = list(range(len(order)))[::-1]
    ax.barh(ys, [cnt[k] for k in order], 0.55, color=BLUE, edgecolor="white", linewidth=1.5)
    n = len(rows)
    for y, k in zip(ys, order):
        ax.text(cnt[k] + 0.6, y, f"{cnt[k]} ({100 * cnt[k] / n:.0f}%)", va="center", fontsize=7, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([REASON_LABEL.get(k, k) for k in order], fontsize=7.5)
    ax.set_xlabel(f"real detections rejected by the VLM (n = {n})"); ax.set_xlim(0, max(cnt.values()) * 1.25)
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_reasons")


def fig_examples(ids):
    set_dir = Path("vlm_sets") / MAIN
    ans_path = Path("vlm_runs") / TAG / "answers.csv"
    if not (set_dir / "items.csv").exists() or not ans_path.exists():
        print("  (VLM files for the main run not found: Fig 8 skipped)")
        return
    items = {r["item_id"]: r for r in csv.DictReader(open(set_dir / "items.csv", encoding="utf-8"))}
    ans = {r["item_id"]: r["answer"] for r in csv.DictReader(open(ans_path, encoding="utf-8"))}
    reasons = {}
    bp = Path("vlm_results/hand_check/B_check.csv")
    if bp.exists():
        reasons = {r["item_id"]: r["reason"].strip().lower() for r in csv.DictReader(open(bp, encoding="utf-8-sig"))}
    if ids:
        chosen = [i.strip() for i in ids.split(",") if i.strip()]
        titles = ["" for _ in chosen]
    else:
        def best(f):
            c = sorted((it for i, it in items.items() if i in ans and f(it, ans[i])), key=lambda it: -float(it["yolo_score"]))
            return c[0]["item_id"] if c else None
        picks = [(best(lambda it, a: it["set"] == "A" and it["kind"] == "hard_absent" and a == "no"), "lookalike, removed by check"),
                 (best(lambda it, a: it["set"] == "A" and it["kind"] == "hard_absent" and a == "yes"), "lookalike, kept by check"),
                 (best(lambda it, a: it["set"] == "B" and a == "no" and reasons.get(it["item_id"]) == "hand"), "real 'person', rejected (hand only)"),
                 (best(lambda it, a: it["set"] == "B" and a == "yes"), "real object, kept by check")]
        chosen = [p for p, _ in picks if p]
        titles = [t for p, t in picks if p]
    fig, axs = plt.subplots(1, len(chosen), figsize=(1.85 * len(chosen), 2.6))
    axs = np.atleast_1d(axs)
    for ax, iid, t in zip(axs, chosen, titles):
        it = items[iid]
        img = plt.imread(set_dir / "crops" / f"{iid}.jpg")
        ax.imshow(img); ax.axis("off")
        f = float(it["saved_w"]) / float(it["crop_w"])
        x1 = (float(it["x1"]) - float(it["crop_x1"])) * f; y1 = (float(it["y1"]) - float(it["crop_y1"])) * f
        x2 = (float(it["x2"]) - float(it["crop_x1"])) * f; y2 = (float(it["y2"]) - float(it["crop_y1"])) * f
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ffd400", lw=1.4))
        on = f"\non: {it['sits_on']}" if it["set"] == "A" else ""
        ax.set_title(f"{t}\nword: '{it['prompt']}'{on}\nVLM: {ans.get(iid, '?')}", fontsize=6.6, color=INK)
        print(f"  Fig 8 panel: {iid} ({t or 'chosen by you'})")
    save(fig, "fig8_examples")


# ===================================================================== table, numbers, check
def write_table(scores):
    head = ["setting", "ms_per_frame_prompt", "found_in_video", "frame_hit", "unrelated_FA", "lookalike_FA",
            "strict_lookalike_FA", "misnaming_share_of_lookalike_FA", "lookalike_FA_after_VLM"]
    rows = []
    for (s, r), run in SEP.items():
        if run not in scores:
            continue
        rs = scores[run]
        hard = sel(rs, "hard")
        fired = [v for v in hard.values() if v["fired"]]
        mis = 100 * sum(v["mis"] for v in fired) / len(fired) if fired else float("nan")
        ms = speed_ms(run)
        vals = [f"{s} {r}", ms, mean_of(sel(rs, "present"), "found"), mean_of(sel(rs, "present"), "hit"),
                mean_of(sel(rs, "easy"), "fired"), mean_of(hard, "fired"), mean_of(sel(rs, "strict"), "fired"), mis,
                NUMBERS.get(("vlm_after", s, r))]
        for key, v in zip(("ms", "found", "frame_hit", "easy", "hard", "strict", "misnaming_share"), vals[1:8]):
            if v is not None:
                note((key, s, r), v)
        rows.append(vals)
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.1f}" if isinstance(v, float) else str(v))
    with open(OUT / "table1_main.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(head); w.writerows([[fmt(x) for x in r] for r in rows])
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(fmt(x) for x in r) + " |" for r in rows]
    (OUT / "table1_main.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {OUT}/table1_main.csv / .md")


def write_numbers_and_check():
    with open(OUT / "numbers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["measure", "size", "resolution", "value"])
        for (m, s, r), v in sorted(NUMBERS.items(), key=lambda kv: [str(x) for x in kv[0]]):
            w.writerow([m, s, r, round(v, 4) if isinstance(v, float) else v])
    L = ["# Check: recomputed from raw files vs first draft\n",
         "Tolerance 0.15 points (draft values were rounded to 0.1). Bootstrap intervals use a new seed, so they may differ "
         "in the last digit; only the centre values are checked.\n",
         "| measure | size | res | draft | recomputed | status |", "|---|---|---|---:|---:|---|"]
    bad = 0
    for key, d in DRAFT.items():
        v = NUMBERS.get(key)
        if v is None:
            st = "not computed (file missing)"
        elif abs(v - d) <= 0.15:
            st = "OK"
        else:
            st = "DIFF"; bad += 1
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {d} | {'-' if v is None else f'{v:.2f}'} | {st} |")
    L += ["", f"**{bad} difference(s).**" if bad else "**All recomputed values match the draft.**"]
    (OUT / "check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="scores_full/prompt_scores.csv")
    ap.add_argument("--fig8", default="", help="comma-separated item_ids for the example figure (default: chosen automatically)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    print(f"Reading {a.scores} (565k rows, about a minute) ...", flush=True)
    scores = load_scores(a.scores)
    print(f"  runs found: {len(scores)} of 9")
    fig_pipeline()
    fig_main(scores)
    fig_joint(scores)
    fig_speed(scores)
    fig_frames()
    fig_vlm(scores)
    fig_reasons()
    fig_examples(a.fig8)
    write_table(scores)
    write_numbers_and_check()
    print(f"\nDone. Open the {OUT}/ folder to look at the figures.")


if __name__ == "__main__":
    main()
