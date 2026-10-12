"""
vlm_hand_check.py  --  two small hand checks of the VLM test set (medium 640), and their scoring.   (CPU only)

Reads only: vlm_sets/<run>/items.csv, crops/, and the VLM answers. Writes only into vlm_results/hand_check/.
Nothing else is changed.

  make-a   BLIND random sample of set A (default 100). Set A = boxes the dataset says are a lookalike error
           (e.g. 'duck' on a labelled penguin). Question per crop:
               "Is it acceptable to call the thing in the yellow box a '<prompt>'?"   yes / no / unsure
           The VLM answers and the dataset's label are hidden.
           -> A_check_page1.jpg, A_check_page2.jpg, A_check.csv (fill the 'truth' column)

  make-b   ALL set B items the VLM rejected (real objects it said 'no' to). Sort each by reason:
               hand     the box is a 'person' but shows mainly a hand / arm
               surface  the labelled thing (table, board, bowl, chair...) is covered by another object on/in it
               miss     the object is clearly visible; the VLM is simply wrong
               other    anything else (write a short note in the 'note' column)
           -> B_check.jpg, B_check.csv (fill the 'reason' column)

  score    reads the two filled CSVs and prints:
           A: share of set A that is actually an acceptable name [95% interval], and the VLM's removal rate
              on the true lookalikes only (corrected) vs on all of A (as reported)
           B: count and share per reason
           -> vlm_results/hand_check/hand_check_summary.md

Usage (from the OVT-B-Dataset folder):
  python vlm_hand_check.py make-a
  python vlm_hand_check.py make-b
  python vlm_hand_check.py score
Yellow rectangle = the box YOLO drew. Fill the CSVs in Excel; save as 'CSV (Comma delimited)'.
"""
import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

RUN = "yolov8m-worldv2_img640_fp16_separate"
SET_DIR = Path("vlm_sets") / RUN
ANS = Path("vlm_runs/rtx3050__qwen3-vl-2b-instruct/answers.csv")
OUT = Path("vlm_results/hand_check")
SEED = 20261010


def wilson(k, n, z=1.96):
    if n == 0:
        return "-"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f}% [{100 * max(0, c - h):.1f}, {100 * min(1, c + h):.1f}] (n={n})"


def size_name(a):
    a = float(a)
    return "small" if a < 0.01 else "medium" if a < 0.10 else "large"


def load():
    items = {r["item_id"]: r for r in csv.DictReader(open(SET_DIR / "items.csv", encoding="utf-8"))}
    answers = {r["item_id"]: r["answer"] for r in csv.DictReader(open(ANS, encoding="utf-8"))}
    return items, answers


def sheet(sel, captions, path, tile=230, cols=5):
    rows = (len(sel) + cols - 1) // cols
    cap_h = 44
    img_out = np.full((rows * (tile + cap_h), cols * tile, 3), 255, np.uint8)
    for k, (it, (l1, l2)) in enumerate(zip(sel, captions)):
        r, c = divmod(k, cols)
        y0, x0 = r * (tile + cap_h), c * tile
        im = cv2.imread(str(SET_DIR / "crops" / f"{it['item_id']}.jpg"))
        if im is not None:
            s = min((tile - 6) / im.shape[1], (tile - 6) / im.shape[0])
            im2 = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
            oy, ox = y0 + (tile - im2.shape[0]) // 2, x0 + (tile - im2.shape[1]) // 2
            img_out[oy:oy + im2.shape[0], ox:ox + im2.shape[1]] = im2
            try:
                f = float(it["saved_w"]) / float(it["crop_w"]) * s
                p1 = (ox + int((float(it["x1"]) - float(it["crop_x1"])) * f), oy + int((float(it["y1"]) - float(it["crop_y1"])) * f))
                p2 = (ox + int((float(it["x2"]) - float(it["crop_x1"])) * f), oy + int((float(it["y2"]) - float(it["crop_y1"])) * f))
                cv2.rectangle(img_out, p1, p2, (0, 220, 255), 1)
            except (KeyError, ValueError, ZeroDivisionError):
                pass
        cv2.putText(img_out, l1[:32], (x0 + 4, y0 + tile + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_out, l2[:36], (x0 + 4, y0 + tile + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img_out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def guard(path):
    if path.exists():
        raise SystemExit(f"{path} already exists (it may hold your answers). Rename or delete it first if you "
                         f"really want a new one.")


def make_a(n):
    items, answers = load()
    OUT.mkdir(parents=True, exist_ok=True)
    guard(OUT / "A_check.csv")
    pool = sorted(i for i, it in items.items() if it["set"] == "A" and i in answers)
    sel_ids = sorted(random.Random(SEED).sample(pool, min(n, len(pool))))
    sel = [items[i] for i in sel_ids]
    with open(OUT / "A_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "item_id", "prompt", "truth"])
        for k, it in enumerate(sel, 1):
            w.writerow([k, it["item_id"], it["prompt"], ""])
    for p in range(0, len(sel), 50):
        part = sel[p:p + 50]
        caps = [(f"#{p + k + 1}  '{it['prompt']}'?", f"{it['item_id']} {size_name(it['box_rel_area'])}")
                for k, it in enumerate(part)]
        sheet(part, caps, OUT / f"A_check_page{p // 50 + 1}.jpg")
    print(f"{len(sel)} set-A crops (blind) -> {OUT}/A_check_page*.jpg ; fill 'truth' in {OUT / 'A_check.csv'}")


def make_b():
    items, answers = load()
    OUT.mkdir(parents=True, exist_ok=True)
    guard(OUT / "B_check.csv")
    sel = sorted((it for i, it in items.items() if it["set"] == "B" and answers.get(i) == "no"),
                 key=lambda it: it["item_id"])
    if not sel:
        raise SystemExit("No rejected set-B items found (check the answers file path).")
    with open(OUT / "B_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "item_id", "prompt", "size", "reason", "note"])
        for k, it in enumerate(sel, 1):
            w.writerow([k, it["item_id"], it["prompt"], size_name(it["box_rel_area"]), "", ""])
    caps = [(f"#{k + 1}  '{it['prompt']}' -> no", f"{it['item_id']} {size_name(it['box_rel_area'])}")
            for k, it in enumerate(sel)]
    sheet(sel, caps, OUT / "B_check.jpg")
    print(f"{len(sel)} rejected set-B crops -> {OUT / 'B_check.jpg'} ; fill 'reason' in {OUT / 'B_check.csv'}")


def score():
    items, answers = load()
    L = ["# Hand check of the VLM test set (medium 640)\n"]
    pa = OUT / "A_check.csv"
    if pa.exists():
        rows = [r for r in csv.DictReader(open(pa, encoding="utf-8-sig"))]
        t = {r["item_id"]: r["truth"].strip().lower() for r in rows}
        judged = [i for i, v in t.items() if v in ("yes", "no")]
        acceptable = [i for i in judged if t[i] == "yes"]
        true_err = [i for i in judged if t[i] == "no"]
        unsure = sum(1 for v in t.values() if v == "unsure")
        blank = sum(1 for v in t.values() if v not in ("yes", "no", "unsure"))
        vlm_no = lambda ids: sum(1 for i in ids if answers.get(i) != "yes")
        L += ["## Set A sample (blind)\n",
              f"- sample: {len(rows)}; judged yes/no: {len(judged)}; unsure: {unsure}; blank/other: {blank}",
              f"- acceptable name after all (dataset 'error' is not an error): {wilson(len(acceptable), len(judged))}",
              f"- VLM removal rate on ALL judged sample items (as reported): {wilson(vlm_no(judged), len(judged))}",
              f"- VLM removal rate on TRUE lookalike errors only (corrected): {wilson(vlm_no(true_err), len(true_err))}",
              f"- VLM said yes on the acceptable-name items (agrees with you): "
              f"{wilson(len(acceptable) - vlm_no(acceptable), len(acceptable))}", ""]
    else:
        L.append("- set A sheet not made yet\n")
    pb = OUT / "B_check.csv"
    if pb.exists():
        rows = [r for r in csv.DictReader(open(pb, encoding="utf-8-sig"))]
        reasons = Counter(r["reason"].strip().lower() or "(blank)" for r in rows)
        L += ["## Set B: real objects the VLM rejected, by reason\n", "| reason | count | share |", "|---|---:|---:|"]
        for k, v in reasons.most_common():
            L.append(f"| {k} | {v} | {100 * v / len(rows):.0f}% |")
        n_b = sum(1 for it in items.values() if it["set"] == "B")
        miss = reasons.get("miss", 0)
        L += ["", f"- set B size: {n_b}; rejected: {len(rows)} ({100 * len(rows) / n_b:.1f}%); "
                  f"of which clear VLM misses: {miss} ({100 * miss / n_b:.1f}% of all set B)"]
    else:
        L.append("- set B sheet not made yet")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hand_check_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make-a", "make-b", "score"])
    ap.add_argument("--n", type=int, default=100, help="set A sample size")
    a = ap.parse_args()
    {"make-a": lambda: make_a(a.n), "make-b": make_b, "score": score}[a.step]()


if __name__ == "__main__":
    main()
