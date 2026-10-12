"""Read-only check: does OVT-B have enough verified-absent categories
for an "is the object in this video?" study?

Does NOT run any model or modify any file except writing one report JSON.

Usage (from the OVT-B-Dataset folder):
    python check_negatives.py --ann ovtb_ann.json --out negatives_report.json
"""
import argparse
import json
from collections import Counter, defaultdict
from statistics import median


def load_section(path, key):
    """Stream one top-level list if ijson is installed; else fall back to json.load."""
    try:
        import ijson
        with open(path, "rb") as f:
            return list(ijson.items(f, f"{key}.item"))
    except ImportError:
        if not hasattr(load_section, "_cache"):
            print("ijson not installed; loading whole JSON (needs a few GB RAM)...")
            with open(path, "r", encoding="utf-8") as f:
                load_section._cache = json.load(f)
        return load_section._cache.get(key, [])


def source_of(video):
    name = str(video.get("name", ""))
    return name.split("/")[0] if "/" in name else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--out", default="negatives_report.json")
    args = ap.parse_args()

    videos = load_section(args.ann, "videos")
    tracks = load_section(args.ann, "tracks")
    categories = load_section(args.ann, "categories")

    cat_freq = {c["id"]: c.get("frequency") for c in categories}
    present = defaultdict(set)
    for t in tracks:
        present[t["video_id"]].add(t["category_id"])

    per_source = defaultdict(lambda: Counter())
    neg_counts, overlap_videos = [], 0
    neg_cat_usage = Counter()
    for v in videos:
        src = source_of(v)
        negs = set(v.get("neg_category_ids") or [])
        not_ex = v.get("not_exhaustive_category_ids") or []
        per_source[src]["videos"] += 1
        per_source[src]["with_negatives"] += bool(negs)
        per_source[src]["with_not_exhaustive"] += bool(not_ex)
        if negs:
            neg_counts.append(len(negs))
            neg_cat_usage.update(negs)
            if negs & present[v["id"]]:
                overlap_videos += 1  # data-quality red flag: listed absent but annotated present

    report = {
        "total_videos": len(videos),
        "videos_with_negatives": len(neg_counts),
        "negatives_per_video": {
            "min": min(neg_counts) if neg_counts else 0,
            "median": median(neg_counts) if neg_counts else 0,
            "max": max(neg_counts) if neg_counts else 0,
        },
        "distinct_negative_categories": len(neg_cat_usage),
        "videos_where_negative_conflicts_with_annotation": overlap_videos,
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "category_frequency_labels": dict(Counter(cat_freq.values())),
        "top_20_negative_categories": [
            {"category_id": c, "videos": n} for c, n in neg_cat_usage.most_common(20)
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in list(report)[:6]}, indent=2))
    print("\nPer source:")
    for src, c in report["per_source"].items():
        print(f"  {src:20s} videos={c['videos']:5d}  with_neg={c['with_negatives']:5d}  "
              f"not_exhaustive={c['with_not_exhaustive']:5d}")
    print(f"\nFrequency labels: {report['category_frequency_labels']}")
    print(f"Full report: {args.out}")


if __name__ == "__main__":
    main()
