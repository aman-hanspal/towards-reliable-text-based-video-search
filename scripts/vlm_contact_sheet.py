"""
vlm_contact_sheet.py  --  put many VLM test crops on one picture, each labelled with its prompt and answer.  (CPU)

Use it to SEE what the checker got right or wrong, e.g. the real objects it rejected (set B, answer no).

Each tile: the crop (fitted into a square), and underneath:  item id | set | prompt | VLM answer | box size
For set A the tile also says what the box really sits on ("on: penguin").

Usage (from the OVT-B-Dataset folder):
  python vlm_contact_sheet.py --set B --answer no                    # real objects the VLM rejected
  python vlm_contact_sheet.py --set A --answer yes                   # lookalikes the VLM let through
  python vlm_contact_sheet.py --set B --answer no --size large       # only large boxes
  python vlm_contact_sheet.py --set C --blind                        # ALL set C crops, numbered, answers HIDDEN
  python vlm_contact_sheet.py --set C --answer any                   # set C with answers (only AFTER your blind check)
Yellow rectangle = the box YOLO drew (the crop adds 20% margin around it).
Options: --run (default medium 640 separate), --n tiles (default 40), --seed, --out file.jpg
Output: vlm_results/sheets/<run>_<set>_<answer>[_<size>].jpg  and a .csv listing the same items.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

TAG = "rtx3050__qwen3-vl-2b-instruct"
MAIN = "yolov8m-worldv2_img640_fp16_separate"


def size_name(a):
    a = float(a)
    return "small" if a < 0.01 else "medium" if a < 0.10 else "large"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=MAIN)
    ap.add_argument("--set", choices=["A", "B", "C"], required=True)
    ap.add_argument("--answer", choices=["yes", "no", "any"], default="any")
    ap.add_argument("--size", choices=["small", "medium", "large", "any"], default="any")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--out", default="")
    ap.add_argument("--blind", action="store_true", help="hide the VLM answers (for your own eye check); shows ALL items")
    args = ap.parse_args()

    set_dir = Path("vlm_sets") / args.run
    ans_dir = Path("vlm_runs") / TAG if args.run == MAIN else Path("vlm_runs/by_run") / args.run / TAG
    items = {r["item_id"]: r for r in csv.DictReader(open(set_dir / "items.csv", encoding="utf-8"))}
    answers = {r["item_id"]: r["answer"] for r in csv.DictReader(open(ans_dir / "answers.csv", encoding="utf-8"))}

    sel = [it for iid, it in items.items() if it["set"] == args.set and iid in answers
           and (args.answer == "any" or answers[iid] == args.answer)
           and (args.size == "any" or size_name(it["box_rel_area"]) == args.size)]
    total = len(sel)
    random.Random(args.seed).shuffle(sel)
    sel = sorted(sel if args.blind else sel[:args.n], key=lambda it: it["item_id"])
    if not sel:
        raise SystemExit("No items match.")

    T, cols = args.tile, 5
    rows = (len(sel) + cols - 1) // cols
    cap_h = 46
    sheet = np.full((rows * (T + cap_h), cols * T, 3), 255, np.uint8)
    for k, it in enumerate(sel):
        img = cv2.imread(str(set_dir / "crops" / f"{it['item_id']}.jpg"))
        r, c = divmod(k, cols)
        y0, x0 = r * (T + cap_h), c * T
        if img is not None:
            s = min((T - 6) / img.shape[1], (T - 6) / img.shape[0])
            im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
            oy, ox = y0 + (T - im.shape[0]) // 2, x0 + (T - im.shape[1]) // 2
            sheet[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
            try:  # YOLO's box inside the crop (crop = box + margin), drawn in yellow
                f = (float(it["saved_w"]) / float(it["crop_w"])) * s
                bx1 = ox + int((float(it["x1"]) - float(it["crop_x1"])) * f)
                by1 = oy + int((float(it["y1"]) - float(it["crop_y1"])) * f)
                bx2 = ox + int((float(it["x2"]) - float(it["crop_x1"])) * f)
                by2 = oy + int((float(it["y2"]) - float(it["crop_y1"])) * f)
                cv2.rectangle(sheet, (bx1, by1), (bx2, by2), (0, 220, 255), 1)
            except (KeyError, ValueError, ZeroDivisionError):
                pass
        a = answers[it["item_id"]]
        color = (0, 140, 0) if a == "yes" else (0, 0, 200)
        line1 = f"'{it['prompt']}' -> {a}"
        if args.blind:
            color, line1 = (0, 0, 0), f"#{k + 1}  is it a '{it['prompt']}'?"
        line2 = f"{it['item_id']} {size_name(it['box_rel_area'])}" + (f" on:{it['sits_on']}" if it["set"] == "A" else "")
        cv2.putText(sheet, line1[:30], (x0 + 4, y0 + T + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(sheet, line2[:34], (x0 + 4, y0 + T + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    out_dir = Path("vlm_results/sheets")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_{args.set}_{args.answer}" + ("" if args.size == "any" else f"_{args.size}")
    if args.blind:
        stem = f"{args.run}_{args.set}_BLIND"
    out = Path(args.out) if args.out else out_dir / f"{stem}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if args.blind:
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["number", "item_id", "prompt", "truth"])
            for k, it in enumerate(sel, 1):
                w.writerow([k, it["item_id"], it["prompt"], ""])
        print(f"{len(sel)} items (answers hidden) -> {out}  |  fill 'truth' in {out.with_suffix('.csv')}")
        return
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "prompt", "answer", "size", "sits_on", "video_id", "frame", "yolo_score"])
        for it in sel:
            w.writerow([it["item_id"], it["prompt"], answers[it["item_id"]], size_name(it["box_rel_area"]),
                        it["sits_on"], it["video_id"], it["frame"], it["yolo_score"]])
    print(f"{total} items match; showing {len(sel)} -> {out}")


if __name__ == "__main__":
    main()
