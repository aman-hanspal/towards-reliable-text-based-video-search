"""
first_look.py  --  first analysis of the full-run scores (reads scores_full/prompt_scores.csv, no GPU).

What it prints, all at threshold 0.25 unless stated:
  1. One headline line per run: present frame hit rate, easy / hard / strict-hard "fired" rates.
  2. Paired comparisons between runs (same videos, same prompts), each with a 95% confidence interval
     from a bootstrap over videos (explained below).
  3. For medium 640 separate: how many hard false alarms are misnaming, which prompts fire most,
     and how many present objects are never found even at threshold 0.05.

Bootstrap in one paragraph: we have 1,972 videos. We draw 1,972 videos AT RANDOM WITH REPLACEMENT
(some twice, some not at all), recompute the difference between two runs, and repeat 2,000 times.
The middle 95% of those 2,000 differences is the confidence interval. If it does not include 0,
the difference is unlikely to be luck of which videos happen to be in the dataset.
We resample whole videos (not single prompts) because prompts in the same video are not independent.

Usage:  python first_look.py                 (default file scores_full/prompt_scores.csv)
        python first_look.py --csv other.csv --threshold 0.1
"""
import argparse

import numpy as np
import pandas as pd


def run_name(size, img, mode="separate"):
    return f"yolov8{size}-worldv2_img{img}_fp16_{mode}"


def per_video(df, kind, col):
    """Per video: sum of the metric over its prompts and number of prompts.
    Mean over all prompts = total sum / total count (same averaging as score_summary.md)."""
    x = df[df.kind == kind]
    if kind == "present":
        x = x[x.gt_boxes > 0]
    return x.groupby("video_id")[col].agg(["sum", "count"])


def paired_diff(t, run_a, run_b, kind, col, rng, n_boot=2000):
    a, b = per_video(t[t.run == run_a], kind, col), per_video(t[t.run == run_b], kind, col)
    vids = a.index.intersection(b.index).values
    sa, ca = a.loc[vids, "sum"].values, a.loc[vids, "count"].values
    sb, cb = b.loc[vids, "sum"].values, b.loc[vids, "count"].values
    est = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(vids), len(vids))
        boots.append(sa[i].sum() / ca[i].sum() - sb[i].sum() / cb[i].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * est, 100 * lo, 100 * hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="scores_full/prompt_scores.csv")
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    t = d[d.threshold == args.threshold]
    rng = np.random.default_rng(0)
    print(f"{len(d):,} rows, {d.run.nunique()} runs, threshold {args.threshold}\n")

    print("1. Headline per run")
    print(f"{'run':40s} {'present':>8s} {'easy':>6s} {'hard':>6s} {'strict':>7s}")
    for run in sorted(t.run.unique()):
        r = t[t.run == run]
        pres = r[(r.kind == "present") & (r.gt_boxes > 0)].frame_hit_rate.mean()
        easy = r[r.kind == "easy_absent"].fired.mean()
        hard = r[r.kind == "hard_absent"].fired.mean()
        strict = r[r.strict_hard == True].fired.mean()  # noqa: E712 (pandas column of bools)
        print(f"{run:40s} {100*pres:7.1f}% {100*easy:5.1f}% {100*hard:5.1f}% {100*strict:6.1f}%")

    print("\n2. Paired differences (A minus B, percentage points, [95% interval over videos])")
    comps = [("large vs medium, 640", run_name("l", 640), run_name("m", 640)),
             ("small vs medium, 640", run_name("s", 640), run_name("m", 640)),
             ("480 vs 640, medium", run_name("m", 480), run_name("m", 640)),
             ("480 vs 640, large", run_name("l", 480), run_name("l", 640)),
             ("480 vs 640, small", run_name("s", 480), run_name("s", 640)),
             ("joint vs separate, medium", run_name("m", 640, "joint"), run_name("m", 640)),
             ("joint vs separate, large", run_name("l", 640, "joint"), run_name("l", 640)),
             ("joint vs separate, small", run_name("s", 640, "joint"), run_name("s", 640))]
    for name, a, b in comps:
        cells = []
        for kind, col in (("present", "frame_hit_rate"), ("easy_absent", "fired"), ("hard_absent", "fired")):
            e, lo, hi = paired_diff(t, a, b, kind, col, rng)
            cells.append(f"{kind.split('_')[0]:7s} {e:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")
        print(f"  {name:27s} " + " | ".join(cells))

    main_run = run_name("m", 640)
    m = t[(t.run == main_run) & (t.kind == "hard_absent")]
    print(f"\n3. {main_run}")
    print(f"  hard prompts fired: {int(m.fired.sum())} of {len(m)}; of these, misnaming: "
          f"{int(m.fired_misnaming.sum())} ({100*m.fired_misnaming.sum()/max(1, m.fired.sum()):.0f}%)")
    print("  hard prompts that fire most often:",
          ", ".join(f"{p} ({n})" for p, n in m[m.fired == 1].prompt.value_counts().head(10).items()))
    p = d[(d.run == main_run) & (d.kind == "present") & (d.threshold == 0.05) & (d.gt_boxes > 0)]
    miss = p[p.video_found == 0]
    print(f"  present objects never found even at threshold 0.05: {len(miss)} of {len(p)} "
          f"({100*len(miss)/len(p):.0f}%); from LVVIS: {100*(miss.source == 'LVVIS').mean():.0f}%")


if __name__ == "__main__":
    main()
