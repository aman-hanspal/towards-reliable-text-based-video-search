"""
build_query_set_v3.py  --  freeze the "exam paper" BEFORE any detector runs.

Version 3 changes from v2 (v2 reviewed at sanity check, before any YOLO run):
  * Hard prompts need WordNet cousin AND to be among the anchor's 20 most CLIP-similar candidates
    (was 60): stops wrong-word-sense cousins such as triangle/snake or wheel/drum.
  * Categories resolved only by head noun ('approximate') can no longer be hard prompts.
  * Optional manual_exclusions.csv for documented unfair pairs (e.g. remote control car / toy).

Version 2. Changes from v1 (v1 was rejected at the sanity check, before any YOLO run):
  * OVT-B synset fields are all 'unknown', so categories are looked up in WordNet BY NAME.
    Only physical-object meanings are kept; multi-word names fall back to their head noun.
  * Parent/child exclusion is conservative: ANY meaning related to ANY meaning -> excluded.
  * Hard-absent prompts must be WordNet "cousins" (Wu-Palmer >= --wup-min, not parent/child),
    ranked by CLIP similarity. If none exist the prompt is marked 'clip_fallback'.
  * Each hard prompt is anchored to a different present category in turn, so 'person'
    cannot dominate.
  * Name rule widened: any whole-word overlap ('remote control' / 'remote control car').
  * Loud failures: the script stops if WordNet cannot resolve enough categories.

For every successfully converted OVT-B video it writes:
  present queries      : every annotated category in that video
  easy-absent queries  : unrelated categories NOT annotated in the video
  hard-absent queries  : lookalike categories NOT annotated in the video

Read-only with respect to the dataset. Runs no detector.

Usage (from the OVT-B-Dataset folder):
    python build_query_set_v3.py --root . --out query_set_v3
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

VERSION = 3
DEFAULTS = dict(
    n_easy=2,
    n_hard=2,
    hard_pool=3,            # pick randomly among the 3 most CLIP-similar cousins of an anchor
    cousin_search=20,       # look for cousins among an anchor's 20 most CLIP-similar candidates
    wup_min=0.75,           # Wu-Palmer similarity needed to count as a "cousin"
    easy_quantile=0.5,      # easy prompts come from the least-similar 50% of candidates
    synonym_sim=0.92,       # CLIP sim above this to a present category -> treated as synonym, excluded
    min_wordnet_coverage=0.6,
    seed=20261001,
    clip_model="ViT-B/32",
    prompt_template="a photo of a {}",
)
REFERENCE_PAIRS = [("goose", "duck"), ("dolphin", "seal"), ("car", "truck"), ("bowl", "pot"),
                   ("dog", "bear"), ("dog", "cat"), ("person", "cat"), ("person", "doll"),
                   ("bottle", "alcohol"), ("knob", "ball")]


# ----------------------------------------------------------------- helpers
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_section(path, key):
    try:
        import ijson
    except ImportError:
        if not hasattr(iter_section, "_cache"):
            print("  ijson not installed -> loading whole JSON into RAM (pip install ijson to stream)")
            with open(path, "r", encoding="utf-8") as f:
                iter_section._cache = json.load(f)
        yield from iter_section._cache.get(key, [])
        return
    with open(path, "rb") as f:
        yield from ijson.items(f, f"{key}.item", use_float=True)


def to_prompt(name):
    text = re.sub(r"_?\(.*?\)", "", name).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def words_overlap(a, b):
    """True if the shorter prompt appears as whole words inside the longer one."""
    wa, wb = a.split(), b.split()
    if len(wa) > len(wb):
        wa, wb = wb, wa
    n = len(wa)
    return any(wb[i:i + n] == wa for i in range(len(wb) - n + 1))


# ----------------------------------------------------------------- CLIP
def embed_prompts(prompts, model_name, template, mode):
    import numpy as np
    if mode == "random":
        rng = np.random.default_rng(0)
        v = rng.normal(size=(len(prompts), 64))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    try:
        import clip
        import torch
    except ImportError:
        sys.exit("CLIP not importable (pip install git+https://github.com/ultralytics/CLIP.git)")
    model, _ = clip.load(model_name, device="cpu")
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            toks = clip.tokenize([template.format(p) for p in prompts[i:i + 256]])
            e = model.encode_text(toks).float()
            out.append(e / e.norm(dim=1, keepdim=True))
    return torch.cat(out).numpy()


# ----------------------------------------------------------------- WordNet
class WordNetIndex:
    def __init__(self, wn):
        self.wn = wn
        self._wup = {}

    def _physical(self, synsets):
        phys = [s for s in synsets
                if any(h.name() == "physical_entity.n.01" for p in s.hypernym_paths() for h in p)]
        return phys or synsets

    def resolve(self, name):
        """-> (method, [synsets])  method in {'exact','head','unresolved'}"""
        base = re.sub(r"_?\(.*?\)", "", name).strip("_ ").lower().replace("_", " ")
        forms = [base.replace(" ", "_"), base.replace("-", "_").replace(" ", "_"),
                 base.replace(" ", "").replace("-", "")]
        for form in dict.fromkeys(forms):
            ss = self.wn.synsets(form, pos="n")
            if ss:
                return "exact", self._physical(ss)
        head = re.split(r"[ \-]", base)[-1]
        ss = self.wn.synsets(head, pos="n") if head else []
        if ss:
            return "head", self._physical(ss)
        return "unresolved", []

    @staticmethod
    def ancestors(synsets):
        return {h.name() for s in synsets for p in s.hypernym_paths() for h in p}

    def wup(self, key, ss1, ss2):
        if key not in self._wup:
            best = 0.0
            for a in ss1:
                for b in ss2:
                    v = a.wup_similarity(b)
                    if v and v > best:
                        best = v
            self._wup[key] = best
        return self._wup[key]


def load_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
        return wn
    except Exception as e:
        sys.exit(f"WordNet is required for v2 and could not be loaded ({e}).\n"
                 f"Run: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ann", default="ovtb_ann.json")
    ap.add_argument("--videos-dir", default="prepared_videos")
    ap.add_argument("--out", default="query_set_v3")
    ap.add_argument("--manual-exclusions", default="manual_exclusions.csv",
                    help="CSV: present_prompt,candidate_prompt,reason (applied both ways)")
    ap.add_argument("--embedder", choices=["clip", "random"], default="clip",
                    help="'random' is for testing the script only; never use it for the study")
    ap.add_argument("--allow-low-coverage", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    import numpy as np
    root = Path(args.root)
    ann_path = root / args.ann
    out_dir = root / args.out
    if (out_dir / "query_set.json").exists():
        sys.exit(f"{out_dir/'query_set.json'} already exists. A frozen set is never overwritten; "
                 f"use a new --out folder.")
    out_dir.mkdir(parents=True, exist_ok=True)

    wn = load_wordnet()
    wni = WordNetIndex(wn)

    print("[1/7] Hashing annotation file ...")
    ann_sha = sha256_file(ann_path)

    print("[2/7] Reading converted-video manifests ...")
    usable, skipped = {}, []
    for mf in sorted((root / args.videos_dir).glob("*/*/*.manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        mp4 = mf.parent / m["conversion"]["video_file"]
        problems = []
        if m.get("status") != "complete":
            problems.append(f"status={m.get('status')}")
        if not mp4.exists():
            problems.append("mp4 missing")
        if m["source"].get("annotation_sha256") != ann_sha:
            problems.append("manifest built from a different annotation file")
        if problems:
            skipped.append({"sequence": m.get("sequence"), "problems": problems})
            continue
        usable[m["video_id"]] = {
            "sequence": m["sequence"],
            "mp4": str(mp4.relative_to(root)).replace("\\", "/"),
            "frame_count": m["conversion"]["frame_count"],
            "width": m["source"]["video_record"]["width"],
            "height": m["source"]["video_record"]["height"],
            "manifest_sha256": sha256_file(mf),
        }
    print(f"      usable videos: {len(usable)}   skipped: {len(skipped)}")

    print("[3/7] Reading categories, tracks and annotations (streaming) ...")
    cats = {}
    for c in iter_section(ann_path, "categories"):
        cats[c["id"]] = {"name": c["name"], "prompt": to_prompt(c["name"]),
                         "split": {"b": "base", "r": "novel"}.get(c.get("frequency"), str(c.get("frequency")))}
    present = defaultdict(set)
    for t in iter_section(ann_path, "tracks"):
        if t["video_id"] in usable:
            present[t["video_id"]].add(t["category_id"])
    frames, areas = defaultdict(set), defaultdict(list)
    for a in iter_section(ann_path, "annotations"):
        v = a["video_id"]
        if v not in usable or a.get("iscrowd", 0):
            continue
        key = (v, a["category_id"])
        frames[key].add(a["image_id"])
        info = usable[v]
        areas[key].append(float(a["area"]) / (info["width"] * info["height"]))

    print("[4/7] Resolving category names in WordNet ...")
    method_counts = Counter()
    for cid, c in cats.items():
        method, ss = wni.resolve(c["name"])
        c["wn_method"], c["wn_synsets"] = method, ss
        c["wn_names"] = {s.name() for s in ss}
        c["wn_anc"] = wni.ancestors(ss)
        method_counts[method] += 1
    coverage = (method_counts["exact"] + method_counts["head"]) / max(1, len(cats))
    print(f"      exact {method_counts['exact']}, head-noun {method_counts['head']}, "
          f"unresolved {method_counts['unresolved']}  (coverage {coverage:.1%})")
    with open(out_dir / "category_wordnet_map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category_id", "name", "prompt", "split", "method", "synsets"])
        for cid in sorted(cats):
            c = cats[cid]
            w.writerow([cid, c["name"], c["prompt"], c["split"], c["wn_method"], " ".join(sorted(c["wn_names"]))])
    if coverage < args.min_wordnet_coverage and not args.allow_low_coverage:
        sys.exit(f"WordNet coverage {coverage:.1%} is below {args.min_wordnet_coverage:.0%}. "
                 f"Inspect {out_dir/'category_wordnet_map.csv'} and send it to Claude.")

    def related(a, b):  # parent/child (or same meaning) under ANY sense
        ca, cb = cats[a], cats[b]
        return bool(ca["wn_names"] & cb["wn_anc"]) or bool(cb["wn_names"] & ca["wn_anc"])

    def wup(a, b):
        key = (a, b) if a < b else (b, a)
        return wni.wup(key, cats[a]["wn_synsets"], cats[b]["wn_synsets"])

    reference = []
    for x, y in REFERENCE_PAIRS:
        _, sx = wni.resolve(x)
        _, sy = wni.resolve(y)
        anc_rel = bool({s.name() for s in sx} & wni.ancestors(sy)) or bool({s.name() for s in sy} & wni.ancestors(sx))
        reference.append((x, y, round(wni.wup((x, y), sx, sy), 3), anc_rel))

    print(f"[5/7] Embedding {len(cats)} category names with {args.embedder} ...")
    cat_ids = sorted(cats)
    index = {cid: i for i, cid in enumerate(cat_ids)}
    emb = embed_prompts([cats[c]["prompt"] for c in cat_ids], args.clip_model, args.prompt_template, args.embedder)
    sim = emb @ emb.T

    manual, manual_path = set(), root / args.manual_exclusions
    manual_sha = None
    if manual_path.exists():
        manual_sha = sha256_file(manual_path)
        with open(manual_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a, b = row["present_prompt"].strip().lower(), row["candidate_prompt"].strip().lower()
                manual |= {(a, b), (b, a)}
        print(f"      manual exclusions loaded: {len(manual)//2} pairs from {manual_path.name}")
    else:
        print("      no manual_exclusions.csv found (none applied)")

    print("[6/7] Building queries per video ...")
    filt = Counter()
    hard_methods = Counter()
    anchor_use = Counter()
    videos_out = []
    for vid in sorted(usable):
        info = usable[vid]
        pres = sorted(present.get(vid, set()))
        if not pres:
            filt["videos_without_present_categories"] += 1
            continue
        rng = random.Random(f"{args.seed}:{vid}")
        pres_prompts = {cats[c]["prompt"] for c in pres}

        valid = []
        for c in cat_ids:
            if c in present[vid]:
                continue
            p = cats[c]["prompt"]
            if p in pres_prompts:
                filt["same_prompt_text"] += 1
                continue
            if any((q, p) in manual for q in pres_prompts):
                filt["manual_exclusion"] += 1
                continue
            if any(words_overlap(p, q) for q in pres_prompts):
                filt["whole_word_overlap"] += 1
                continue
            s_max = max(float(sim[index[c], index[q]]) for q in pres)
            if s_max > args.synonym_sim:
                filt["near_synonym_by_clip"] += 1
                continue
            if any(related(c, q) for q in pres):
                filt["wordnet_parent_child_or_same"] += 1
                continue
            valid.append(c)
        valid_set = set(valid)

        # ---- hard: rotate anchors, require WordNet cousins
        anchors = pres[:]
        rng.shuffle(anchors)
        chosen_hard = []
        for slot in range(args.n_hard):
            picked = None
            for k in range(len(anchors)):
                a = anchors[(slot + k) % len(anchors)]
                if not cats[a]["wn_synsets"]:
                    continue
                near = sorted((c for c in valid_set if c not in {h["category_id"] for h in chosen_hard}),
                              key=lambda c: -float(sim[index[a], index[c]]))[:args.cousin_search]
                cousins = [c for c in near if cats[c]["wn_method"] == "exact" and wup(a, c) >= args.wup_min]
                if cousins:
                    c = rng.choice(cousins[:args.hard_pool])
                    picked = {"category_id": c, "anchor_id": a, "method": "wordnet_cousin",
                              "wup": round(wup(a, c), 3)}
                    break
            if picked is None:  # no cousin anywhere in this video -> CLIP fallback, flagged
                taken = {h["category_id"] for h in chosen_hard}
                pool = sorted((c for c in valid_set if c not in taken),
                              key=lambda c: -max(float(sim[index[c], index[q]]) for q in pres))[:args.hard_pool]
                if pool:
                    c = rng.choice(pool)
                    a = max(pres, key=lambda q: float(sim[index[c], index[q]]))
                    picked = {"category_id": c, "anchor_id": a, "method": "clip_fallback",
                              "wup": round(wup(a, c), 3) if cats[a]["wn_synsets"] and cats[c]["wn_synsets"] else None}
            if picked:
                chosen_hard.append(picked)
                hard_methods[picked["method"]] += 1
                anchor_use[cats[picked["anchor_id"]]["prompt"]] += 1

        # ---- easy: least-similar half of the valid candidates
        hard_ids = {h["category_id"] for h in chosen_hard}
        ranked = sorted((c for c in valid if c not in hard_ids),
                        key=lambda c: -max(float(sim[index[c], index[q]]) for q in pres))
        easy_pool = ranked[int(len(ranked) * (1 - args.easy_quantile)):]
        easy_ids = rng.sample(easy_pool, min(args.n_easy, len(easy_pool)))

        def entry(c, extra=None):
            nearest = max(pres, key=lambda q: float(sim[index[c], index[q]]))
            d = {"category_id": c, "name": cats[c]["name"], "prompt": cats[c]["prompt"],
                 "split": cats[c]["split"], "wordnet_method": cats[c]["wn_method"],
                 "max_sim_to_present": round(float(sim[index[c], index[nearest]]), 4),
                 "most_similar_present_id": nearest}
            if extra:
                d.update({"anchor_id": extra["anchor_id"], "anchor_prompt": cats[extra["anchor_id"]]["prompt"],
                          "selection": extra["method"], "wup_to_anchor": extra["wup"],
                          "clip_sim_to_anchor": round(float(sim[index[c], index[extra["anchor_id"]]]), 4)})
            return d

        present_entries = []
        for c in pres:
            fr, ar = frames.get((vid, c), set()), areas.get((vid, c), [])
            present_entries.append({
                "category_id": c, "name": cats[c]["name"], "prompt": cats[c]["prompt"],
                "split": cats[c]["split"], "wordnet_method": cats[c]["wn_method"],
                "annotated_frames": len(fr), "boxes": len(ar),
                "median_rel_box_area": round(median(ar), 6) if ar else None})

        videos_out.append({
            "video_id": vid, **{k: info[k] for k in ("sequence", "mp4", "frame_count", "width", "height",
                                                    "manifest_sha256")},
            "source": info["sequence"].split("/")[0],
            "present": present_entries,
            "easy_absent": [entry(c) for c in easy_ids],
            "hard_absent": [entry(h["category_id"], h) for h in chosen_hard],
        })

    print("[7/7] Writing outputs ...")
    meta = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen query set for open-vocabulary video search evaluation. Do not edit.",
        "annotation_sha256": ann_sha,
        "script_sha256": sha256_file(__file__),
        "embedder": args.embedder,
        "parameters": {k: getattr(args, k) for k in DEFAULTS},
        "wordnet_resolution": dict(method_counts),
        "wordnet_coverage": round(coverage, 4),
        "manual_exclusions_sha256": manual_sha,
        "manual_exclusion_pairs": len(manual) // 2,
        "head_noun_categories": sorted(cats[c]["prompt"] for c in cats if cats[c]["wn_method"] == "head"),
        "unresolved_categories": sorted(cats[c]["prompt"] for c in cats if cats[c]["wn_method"] == "unresolved"),
        "reference_pairs": [{"a": a, "b": b, "wup": w, "parent_child": r} for a, b, w, r in reference],
        "hard_selection_methods": dict(hard_methods),
        "videos_usable": len(usable),
        "videos_in_set": len(videos_out),
        "videos_skipped": skipped,
        "filter_counts": dict(filt),
    }
    qs_path = out_dir / "query_set.json"
    with open(qs_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "videos": videos_out}, f, indent=1)
    qs_sha = sha256_file(qs_path)
    write_summary(out_dir, meta, videos_out, qs_sha, anchor_use)
    write_sanity_sample(out_dir, videos_out, args.seed)
    print(f"\nDone. {len(videos_out)} videos.\nquery_set.json SHA-256: {qs_sha}\n"
          f"Send Claude: {out_dir/'query_set_summary.md'} and {out_dir/'sanity_sample.md'}")


def write_summary(out_dir, meta, videos, qs_sha, anchor_use):
    kinds = {k: [q for v in videos for q in v[k]] for k in ("present", "easy_absent", "hard_absent")}
    L = [f"# Query set summary (v{meta['version']})\n", f"- query_set.json SHA-256: `{qs_sha}`",
         f"- embedder: **{meta['embedder']}**",
         f"- videos in set: {meta['videos_in_set']} (usable {meta['videos_usable']}, "
         f"skipped {len(meta['videos_skipped'])})", ""]
    if meta["embedder"] != "clip":
        L.append("> **WARNING: not built with CLIP. Test output only, not for the study.**\n")
    r = meta["wordnet_resolution"]
    L += ["## WordNet resolution of category names\n",
          f"- exact name: {r.get('exact', 0)} | head noun only (approximate): {r.get('head', 0)} | "
          f"unresolved: {r.get('unresolved', 0)} | coverage {meta['wordnet_coverage']:.1%}",
          "- full mapping: `category_wordnet_map.csv`",
          f"- head-noun (approximate, never used as hard prompts): {', '.join(meta['head_noun_categories'])}",
          f"- unresolved: {', '.join(meta['unresolved_categories'])}",
          f"- manual exclusion pairs: {meta['manual_exclusion_pairs']} (sha256 {meta['manual_exclusions_sha256']})\n",
          "## Reference pairs (is the cousin threshold sensible?)\n",
          f"Threshold: Wu-Palmer >= {meta['parameters']['wup_min']}. Parent/child pairs are always excluded.\n",
          "| pair | Wu-Palmer | parent/child? |", "|---|---:|---|"]
    L += [f"| {p['a']} / {p['b']} | {p['wup']} | {p['parent_child']} |" for p in meta["reference_pairs"]]
    L += ["", "## How hard prompts were chosen\n"]
    L += [f"- {k}: {v}" for k, v in sorted(meta["hard_selection_methods"].items())]
    L += ["", "## Query counts\n", "| type | total | base | novel |", "|---|---:|---:|---:|"]
    for k, qs in kinds.items():
        s = Counter(q["split"] for q in qs)
        L.append(f"| {k} | {len(qs)} | {s.get('base', 0)} | {s.get('novel', 0)} |")
    L += ["", "## Diversity check: most frequent prompts (one word dominating = bad)\n"]
    for k in ("hard_absent", "easy_absent"):
        top = Counter(q["prompt"] for q in kinds[k]).most_common(15)
        L.append(f"- **{k}**: " + ", ".join(f"{p} ({n})" for p, n in top))
    L.append("- **hard anchors**: " + ", ".join(f"{p} ({n})" for p, n in anchor_use.most_common(10)))
    short = sum(1 for v in videos if len(v["hard_absent"]) < 2 or len(v["easy_absent"]) < 2)
    L += ["", f"- videos with fewer than 2+2 absent prompts: {short}", "",
          "## Candidates removed by fairness filters (summed over videos)\n"]
    L += [f"- {k}: {c}" for k, c in sorted(meta["filter_counts"].items())]
    if meta["videos_skipped"]:
        L += ["", "## Skipped videos\n"] + [f"- {s['sequence']}: {', '.join(s['problems'])}"
                                            for s in meta["videos_skipped"]]
    (out_dir / "query_set_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_sanity_sample(out_dir, videos, seed):
    rng = random.Random(f"{seed}:sanity")
    sample = rng.sample(videos, min(30, len(videos)))
    L = ["# Sanity sample (v3): 30 random videos\n",
         "For each HARD-absent prompt ask: *could this prompt honestly describe something that is in the "
         "video, or be its parent/child?* If yes -> UNFAIR. A fair prompt for an object that might be "
         "present but unlabelled is NOT unfair; that is caught later at verify-on-fire.\n"]
    for v in sample:
        L.append(f"## {v['sequence']}  (video_id {v['video_id']}, {v['frame_count']} frames)")
        L.append("- present: " + ", ".join(f"{q['prompt']} [{q['split']}]" for q in v["present"]))
        L.append("- easy absent: " + ", ".join(q["prompt"] for q in v["easy_absent"]))
        L.append("- hard absent: " + "; ".join(
            f"{q['prompt']} (anchor '{q['anchor_prompt']}', {q['selection']}, wup {q['wup_to_anchor']})"
            for q in v["hard_absent"]))
        L.append("- your verdict: fair / unfair because ...\n")
    (out_dir / "sanity_sample.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()