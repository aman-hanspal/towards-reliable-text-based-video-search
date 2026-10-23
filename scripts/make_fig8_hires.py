"""
make_fig8_hires.py  --  redraw Fig. 8 (example crops) at print resolution, with (a)-(d) panel labels.   (CPU only)

Why: make_figures.py let matplotlib resample the photo crops to the figure's save dpi (100 dpi in the PDF,
220 dpi in the PNG), so the crops looked low-resolution. This script draws the SAME four crops from the
saved crop files without resampling (interpolation="none"), so the PDF keeps every original pixel, and
saves the PNG at 600 dpi.

Reads (only reads):
  vlm_sets/<main run>/items.csv and crops/<item_id>.jpg
  vlm_runs/rtx3050__qwen3-vl-2b-instruct/answers.csv
  vlm_results/hand_check/B_check.csv
Writes:
  paper_figures/fig8_examples.png  (600 dpi)  and  paper_figures/fig8_examples.pdf

Usage (from the OVT-B-Dataset folder / repository root):
  python scripts/make_fig8_hires.py
  python scripts/make_fig8_hires.py --ids A_..,A_..,B_..,B_..      # choose the four crops yourself
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MAIN = "yolov8m-worldv2_img640_fp16_separate"
TAG = "rtx3050__qwen3-vl-2b-instruct"
OUT = Path("paper_figures")
EXPECTED = ["manatee", "beachball", "person", "polar bear"]   # words in the published Fig. 8, panels (a)-(d)
INK = "#0b0b0b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default="", help="comma-separated item_ids (default: same automatic choice as make_figures.py)")
    a = ap.parse_args()

    set_dir = Path("vlm_sets") / MAIN
    items = {r["item_id"]: r for r in csv.DictReader(open(set_dir / "items.csv", encoding="utf-8"))}
    ans = {r["item_id"]: r["answer"] for r in csv.DictReader(open(Path("vlm_runs") / TAG / "answers.csv", encoding="utf-8"))}
    reasons = {r["item_id"]: r["reason"].strip().lower()
               for r in csv.DictReader(open("vlm_results/hand_check/B_check.csv", encoding="utf-8-sig"))}

    if a.ids:
        chosen = [i.strip() for i in a.ids.split(",") if i.strip()]
    else:  # identical rule to make_figures.py: highest YOLO score in each of the four groups
        def best(f):
            c = sorted((it for i, it in items.items() if i in ans and f(it, ans[i])), key=lambda it: -float(it["yolo_score"]))
            return c[0]["item_id"] if c else None
        chosen = [best(lambda it, x: it["set"] == "A" and it["kind"] == "hard_absent" and x == "no"),
                  best(lambda it, x: it["set"] == "A" and it["kind"] == "hard_absent" and x == "yes"),
                  best(lambda it, x: it["set"] == "B" and x == "no" and reasons.get(it["item_id"]) == "hand"),
                  best(lambda it, x: it["set"] == "B" and x == "yes")]
    words = [items[i]["prompt"] for i in chosen]
    print("Chosen crops:", ", ".join(f"{i} ('{w}')" for i, w in zip(chosen, words)))
    if not a.ids and words != EXPECTED:
        raise SystemExit(f"STOP: expected the words {EXPECTED} (as in the paper) but got {words}. "
                         "Tell Claude before using this figure, or pass --ids.")

    fig, axs = plt.subplots(1, len(chosen), figsize=(1.85 * len(chosen), 2.75))
    for k, (ax, iid) in enumerate(zip(axs, chosen)):
        it = items[iid]
        img = plt.imread(set_dir / "crops" / f"{iid}.jpg")
        ax.imshow(img, interpolation="none")     # no resampling: the PDF keeps every original pixel
        ax.set_anchor("S")                       # all photos share one bottom edge, so the labels line up
        ax.axis("off")
        f = float(it["saved_w"]) / float(it["crop_w"])
        x1 = (float(it["x1"]) - float(it["crop_x1"])) * f; y1 = (float(it["y1"]) - float(it["crop_y1"])) * f
        x2 = (float(it["x2"]) - float(it["crop_x1"])) * f; y2 = (float(it["y2"]) - float(it["crop_y1"])) * f
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ffd400", lw=1.4))
        on = f"\non: {it['sits_on']}" if it["set"] == "A" else ""
        ax.set_title(f"word: '{it['prompt']}'{on}\nVLM: {ans.get(iid, '?')}", fontsize=6.6, color=INK)
        print(f"  ({'abcd'[k]}) {iid}: crop {img.shape[1]}x{img.shape[0]} px")
    fig.canvas.draw()                            # final image positions, so all labels share one baseline
    bottom = min(ax.get_position().y0 for ax in axs)
    for k, ax in enumerate(axs):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, bottom - 0.04, f"({'abcd'[k]})", ha="center", va="top",
                 fontsize=8, fontweight="bold", color=INK)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig8_examples.png", dpi=600)
    fig.savefig(OUT / "fig8_examples.pdf")
    plt.close(fig)
    print(f"Wrote {OUT}/fig8_examples.png (600 dpi) and {OUT}/fig8_examples.pdf")


if __name__ == "__main__":
    main()
