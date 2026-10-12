"""
score_vlm.py  --  how good is the VLM checker?  (no GPU, standard library only)

Reads the frozen test set (items.csv, with the correct answers), one or more VLM runs (answers.csv from
verify_vlm.py), and the detector scores (prompt_scores.csv) for the "before / after" numbers.

Rules (fixed before seeing results):
  - pilot items are excluded from every accuracy number (they only checked the answer format)
  - an answer that is not clearly "yes" (i.e. "unclear") counts as "no": the detection is rejected
  - 95% intervals are Wilson intervals for a proportion

For every VLM run it reports:
  1. Set A (lookalike errors, right answer "no"):  REMOVED = share answered "no"
  2. Set B (real detections, right answer "yes"):  LOST    = share answered "no"
     both also split by easy/hard, strict-hard, base/novel, object size
  3. Set C (unlabelled area): share answered "yes"; and, once you have filled in c_eye_check.csv,
     agreement with your eye check
  4. Before -> after at the detector threshold of the test set (from FROZEN.md, normally 0.25):
       hard / easy false-alarm rate:  rate x [share misnaming x kept_A + share unlabelled x kept_C]
       present "found in video" rate: rate x kept_B
  5. The free alternative: a stricter YOLO threshold that loses the SAME share of real detections.
     How many lookalike false alarms does that remove? (uses each item's YOLO score)
  6. Time per crop (wall seconds): median, mean, 90th percentile; how much of the model was on the GPU
  7. If several runs use the same model: how often they give the same answer (device agreement)

Writes <out>/vlm_summary.md and, the first time, <out>/c_eye_check.csv: a BLIND list of the Set C crops
(no VLM answers in it). Open each crop, write yes / no / unsure in the 'truth' column
("is the main object really a <prompt>?"), save, and run this script again.

Usage:
  python score_vlm.py --set-dir vlm_sets/yolov8m-worldv2_img640_fp16_separate --runs vlm_runs/rtx3050__qwen3-vl-2b
  python score_vlm.py --set-dir ... --runs vlm_runs/rtx3050__qwen3-vl-2b vlm_runs/m1__qwen3-vl-2b
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return "-" if p is None else f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})"


def size_bin(a):
    a = float(a)
    return "small (<1%)" if a < 0.01 else "medium (1-10%)" if a < 0.10 else "large (>10%)"


def load_items(set_dir):
    items = {r["item_id"]: r for r in csv.DictReader(open(set_dir / "items.csv", encoding="utf-8"))}
    frozen = (set_dir / "FROZEN.md").read_text(encoding="utf-8") if (set_dir / "FROZEN.md").exists() else ""
    m = re.search(r"threshold: ([0-9.]+)", frozen)
    return items, float(m.group(1)) if m else 0.25


def load_rates(scores_csv, run_name, thr):
    """Detector rates for this run at this threshold, straight from prompt_scores.csv."""
    n = defaultdict(int)
    k = defaultdict(int)
    with open(scores_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["run"] != run_name or abs(float(r["threshold"]) - thr) > 1e-9:
                continue
            if r["kind"] == "present":
                if r["gt_boxes"] not in ("", "0"):
                    n["present"] += 1
                    k["present"] += int(r["video_found"])
            else:
                n[r["kind"]] += 1
                k[r["kind"]] += int(r["fired"])
    return n, k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="one or more vlm_runs/<device>__<model> folders")
    ap.add_argument("--scores", default="scores_full/prompt_scores.csv")
    ap.add_argument("--out", default="vlm_results")
    args = ap.parse_args()

    set_dir, out = Path(args.set_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items, thr = load_items(set_dir)
    run_name = set_dir.name
    is_main = lambda it: not it["set"].startswith("pilot_")

    # blind eye-check sheet for set C
    eye_path = out / "c_eye_check.csv"
    if not eye_path.exists():
        with open(eye_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "kind", "prompt", "yolo_score", "crop_file", "truth"])
            for it in sorted((i for i in items.values() if i["set"] == "C"), key=lambda i: i["item_id"]):
                w.writerow([it["item_id"], it["kind"], it["prompt"], it["yolo_score"],
                            f"crops/{it['item_id']}.jpg", ""])
        print(f"Wrote blind eye-check sheet: {eye_path}  (fill the 'truth' column: yes / no / unsure)")
    eye = {r["item_id"]: r["truth"].strip().lower() for r in csv.DictReader(open(eye_path, encoding="utf-8"))
           if r["truth"].strip().lower() in ("yes", "no")}

    n_det, k_det = load_rates(args.scores, run_name, thr) if Path(args.scores).exists() else ({}, {})
    if not n_det:
        print(f"(no detector rates found for {run_name} at {thr} in {args.scores}: before/after section skipped)")

    L = [f"# VLM checker results\n", f"- test set: {run_name}, detector threshold {thr}",
         f"- items: " + ", ".join(f"{s} {sum(1 for i in items.values() if i['set'] == s)}"
                                    for s in sorted({i['set'] for i in items.values()})),
         "- pilot items excluded; 'unclear' answers count as 'no'; [95% Wilson interval]\n"]
    answers_by_run = {}
    for run_dir in map(Path, args.runs):
        meta = json.loads((run_dir / "run_meta.json").read_text())
        ans = {r["item_id"]: r for r in csv.DictReader(open(run_dir / "answers.csv", encoding="utf-8"))}
        answers_by_run[run_dir.name] = (meta, ans)
        rows = [(items[i], a) for i, a in ans.items() if i in items and is_main(items[i])]
        say_no = lambda a: a["answer"] != "yes"

        def block(name, sel, metric_no):
            sub = [(it, a) for it, a in rows if sel(it)]
            k = sum(1 for it, a in sub if say_no(a) == metric_no)
            return f"| {name} | {pct(k, len(sub))} |"

        last = meta.get("runs", [{}])[-1]
        L += [f"## {run_dir.name}\n",
              f"- model {meta['model']} (weights {meta.get('weights_sha256', meta.get('model_digest', '?'))[:12]}), Ollama {meta.get('ollama_version')}, "
              f"device {meta.get('device')}",
              f"- model in memory {last.get('loaded_size_mb')} MB, on GPU {last.get('loaded_on_gpu_mb')} MB "
              f"(share {last.get('share_on_gpu')}); thinking switch {last.get('think_flag')}",
              f"- answers so far: {len(rows)} main items; unclear: {sum(1 for _, a in rows if a['answer'] == 'unclear')}\n",
              "### 1-2. Accuracy on the sets with known answers\n", "| group | share |", "|---|---|"]
        A = lambda it: it["set"] == "A"
        B = lambda it: it["set"] == "B"
        L += [block("**A removed** (said no to a lookalike) - all", A, True),
              block("A removed - hard prompts", lambda it: A(it) and it["kind"] == "hard_absent", True),
              block("A removed - strict-hard", lambda it: A(it) and it["strict_hard"] == "True", True),
              block("A removed - easy prompts", lambda it: A(it) and it["kind"] == "easy_absent", True),
              block("A removed - novel prompt", lambda it: A(it) and it["split"] == "novel", True),
              block("A removed - base prompt", lambda it: A(it) and it["split"] == "base", True),
              block("**B lost** (said no to a real object) - all", B, True)]
        for sb in ("small (<1%)", "medium (1-10%)", "large (>10%)"):
            L.append(block(f"B lost - {sb} object", lambda it, sb=sb: B(it) and size_bin(it["box_rel_area"]) == sb, True))
        L += [block("B lost - novel", lambda it: B(it) and it["split"] == "novel", True),
              block("B lost - base", lambda it: B(it) and it["split"] == "base", True)]

        # ---- set C
        C = [(it, a) for it, a in rows if it["set"] == "C"]
        L += ["", "### 3. Set C (fired where nothing is labelled)\n",
              f"- VLM said yes: {pct(sum(1 for _, a in C if a['answer'] == 'yes'), len(C))}"]
        ce = [(it, a) for it, a in C if it["item_id"] in eye]
        if ce:
            agree = sum(1 for it, a in ce if (a["answer"] == "yes") == (eye[it["item_id"]] == "yes"))
            real = sum(1 for it, _ in ce if eye[it["item_id"]] == "yes")
            L += [f"- your eye check: {len(ce)} judged (yes/no), real unlabelled object in {real}",
                  f"- VLM agrees with your eye check: {pct(agree, len(ce))}"]
        else:
            L.append("- eye check not filled in yet (c_eye_check.csv)")

        # ---- before / after
        if n_det:
            def keep(sel):
                sub = [a for it, a in rows if sel(it)]
                return (sum(1 for a in sub if a["answer"] == "yes") / len(sub)) if sub else None

            L += ["", f"### 4. Before -> after, detector threshold {thr}\n",
                  "| prompt type | before | after VLM check |", "|---|---:|---:|"]
            kb = keep(B)
            if kb is not None and n_det.get("present"):
                r = k_det["present"] / n_det["present"]
                L.append(f"| present: found in video | {100 * r:.1f}% | {100 * r * kb:.1f}% |")
            for kind, label in (("hard_absent", "hard (lookalike) false alarm"), ("easy_absent", "easy (unrelated) false alarm")):
                pool_a = sum(1 for i in items.values() if i["set"] in ("A", "pilot_A") and i["kind"] == kind)
                pool_c = sum(1 for i in items.values() if i["set"] == "C" and i["kind"] == kind)
                ka = keep(lambda it: A(it) and it["kind"] == kind)
                kc = keep(lambda it: it["set"] == "C" and it["kind"] == kind)
                if n_det.get(kind) and pool_a + pool_c:
                    r = k_det[kind] / n_det[kind]
                    kept = (pool_a * (ka if ka is not None else 1) + pool_c * (kc if kc is not None else 1)) / (pool_a + pool_c)
                    L.append(f"| {label} | {100 * r:.1f}% | {100 * r * kept:.1f}% |")
            L.append("\n(after = before x share the VLM keeps; misnaming and unlabelled-area alarms weighted by how many there are)")

            # ---- matched stricter threshold
            lost = 1 - kb if kb is not None else None
            b_scores = sorted(float(it["yolo_score"]) for it, _ in rows if B(it))
            hard_items = [i for i in items.values() if i["kind"] == "hard_absent" and i["set"] in ("A", "pilot_A", "C")]
            if lost is not None and b_scores and n_det.get("hard_absent"):
                target = 1 - lost
                t2 = thr
                for s in [thr] + b_scores:
                    if sum(1 for x in b_scores if x >= s) / len(b_scores) <= target + 1e-12:
                        t2 = s
                        break
                else:
                    t2 = max(b_scores) + 1e-6
                score_of = lambda i: float(i.get("prompt_max_score") or i["yolo_score"])
                still = sum(1 for i in hard_items if score_of(i) >= t2)
                r_hard = k_det["hard_absent"] / n_det["hard_absent"]
                L += ["", "### 5. Same loss of real detections with a stricter YOLO threshold instead\n",
                      f"- VLM loses {100 * lost:.1f}% of real detections; a YOLO threshold of {t2:.3f} loses about the same",
                      f"- hard false-alarm rate at that threshold: {100 * r_hard * still / max(1, len(hard_items)):.1f}% "
                      f"(VLM: see table above; before: {100 * r_hard:.1f}%)"]

        # ---- time
        w = [float(a["wall_s"]) for _, a in rows if a["wall_s"]]
        if w:
            ws = sorted(w)
            L += ["", "### 6. Cost\n",
                  f"- seconds per crop (wall): median {median(w):.2f}, mean {mean(w):.2f}, "
                  f"90th percentile {ws[int(0.9 * (len(ws) - 1))]:.2f}; total {sum(w) / 60:.1f} min for {len(w)} crops"]
        L.append("")

    # ---- device agreement (same model = same weights file)
    ident = lambda m: m.get("weights_sha256") or m.get("model_digest")
    names = list(answers_by_run)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if ident(answers_by_run[a][0]) == ident(answers_by_run[b][0])]
    if pairs:
        L += ["## 7. Same model, different devices: do the answers agree?\n"]
        for a, b in pairs:
            A_, B_ = answers_by_run[a][1], answers_by_run[b][1]
            common = [i for i in A_ if i in B_ and i in items and is_main(items[i])]
            same = sum(1 for i in common if A_[i]["answer"] == B_[i]["answer"])
            L.append(f"- {a} vs {b}: {pct(same, len(common))}")
    (out / "vlm_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nWritten: {out / 'vlm_summary.md'}")


if __name__ == "__main__":
    main()
