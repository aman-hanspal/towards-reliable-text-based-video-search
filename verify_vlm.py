"""
verify_vlm.py  --  ask a local vision-language model (through Ollama) about every crop of a VLM test set.

For each crop in <set-dir>/items.csv it sends ONE question with the crop attached:

    Is the main object in this image a {prompt}? Answer with only one word: yes or no.

and records the raw reply, the parsed answer (yes / no / unclear) and the time taken.
Nothing here knows the right answer; scoring is done afterwards by score_vlm.py.

Fixed settings (the same on every machine; stored in run_meta.json, and a resumed run must match them):
  temperature 0 (no randomness), seed 0, at most 8 output tokens, context 4096 tokens,
  thinking switched off when the model supports it.

Which items:
  --part pilot   only the 20 pilot items (check that the model really answers yes/no)
  --part main    everything except the pilot items (the results)
  --part all     both

Output folder: vlm_runs/<device>__<model>/   (model name with ':' replaced by '-')
  run_meta.json    model name + exact digest, Ollama version, device, question, settings,
                   and how much of the model sat on the GPU (from Ollama's /api/ps)
  answers.csv      one row per item: answer, raw reply, seconds (wall), Ollama's own timings
Re-running the same command resumes: items already answered are skipped.

Needs: Ollama running (http://localhost:11434) and the model pulled (ollama pull qwen3-vl:2b).
Standard library only, so it runs on Windows/WSL and on the Mac without installing anything.

Usage:
  python verify_vlm.py --set-dir vlm_sets/yolov8m-worldv2_img640_fp16_separate --model qwen3-vl:2b --device rtx3050 --part pilot
  python verify_vlm.py --set-dir vlm_sets/yolov8m-worldv2_img640_fp16_separate --model qwen3-vl:2b --device rtx3050 --part main
"""
import argparse
import base64
import csv
import json
import platform
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

QUESTION = "Is the main object in this image a {prompt}? Answer with only one word: yes or no."
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 8, "num_ctx": 4096}
FIELDS = ["item_id", "set", "answer", "raw", "done_reason", "thinking_chars", "wall_s", "total_s", "load_s",
          "prompt_eval_s", "eval_s", "prompt_tokens", "output_tokens", "time_utc"]


def call(host, path, payload=None, timeout=90):
    # 90 s: a normal answer takes ~2 s and a cold model load ~10 s; a hung Ollama is retried after 90 s, not 10 min
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse(text):
    t = text.strip().lower().lstrip("\"'*`([ \n").replace("\n", " ")
    first = t.split(" ")[0].strip(".,!:;\"'*`)") if t else ""
    if first == "yes":
        return "yes"
    if first == "no":
        return "no"
    return "unclear"


def ask(host, model, image_b64, prompt, think_flag):
    payload = {"model": model, "stream": False, "keep_alive": "60m", "options": OPTIONS,
               "messages": [{"role": "user", "content": QUESTION.format(prompt=prompt), "images": [image_b64]}]}
    if think_flag is not None:
        payload["think"] = think_flag
    return call(host, "/api/chat", payload)


def ask_with_retry(host, model, image_b64, prompt, think_flag, tries=8, wait_s=30):
    """Ollama briefly unreachable or busy (server restart, out-of-memory hiccup): wait and retry.
    After `tries` failures (~4 min) stop cleanly; re-running the command resumes where it stopped."""
    for k in range(1, tries + 1):
        try:
            return ask(host, model, image_b64, prompt, think_flag)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            err = str(e)
        print(f"   ! Ollama did not answer ({err}); retry {k}/{tries} in {wait_s}s", flush=True)
        time.sleep(wait_s)
    sys.exit("Ollama kept failing. Nothing is lost: answers so far are saved; run the same command again to resume.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--model", default="qwen3-vl:2b")
    ap.add_argument("--device", required=True, help="short name for this machine, e.g. rtx3050 or m1")
    ap.add_argument("--part", choices=["pilot", "main", "all"], required=True)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="vlm_runs")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new items (0 = no limit)")
    args = ap.parse_args()

    set_dir = Path(args.set_dir)
    items = list(csv.DictReader(open(set_dir / "items.csv", encoding="utf-8")))
    if args.part == "pilot":
        items = [r for r in items if r["set"].startswith("pilot_")]
    elif args.part == "main":
        items = [r for r in items if not r["set"].startswith("pilot_")]
    if not items:
        sys.exit(f"No items for part '{args.part}' in {set_dir / 'items.csv'}.")
    out = Path(args.out) / f"{args.device}__{args.model.replace(':', '-').replace('/', '_')}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- server, model digest, version
    try:
        version = call(args.host, "/api/version")["version"]
        tags = call(args.host, "/api/tags")["models"]
    except (urllib.error.URLError, ConnectionError) as e:
        sys.exit(f"Cannot reach Ollama at {args.host} ({e}). Start it (ollama serve, or open the Ollama app).")
    digest = next((m.get("digest") for m in tags if m["name"] == args.model or m.get("model") == args.model), None)
    if digest is None:
        sys.exit(f"Model {args.model} is not pulled here. Run: ollama pull {args.model}")
    # The manifest digest above can change when Ollama rewrites a model's manifest (seen right after a pull / first
    # load) although the weights are untouched. The stable identity is the WEIGHTS file named in the Modelfile
    # ('FROM .../blobs/sha256-...'): that is what must match between runs.
    weights = None
    try:
        mf = call(args.host, "/api/show", {"model": args.model}).get("modelfile", "")
        for line in mf.splitlines():
            if line.startswith("FROM ") and "sha256" in line:
                weights = line.split("sha256")[-1].lstrip("-:")
                break
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
    if weights is None:
        sys.exit("Could not read the model's weights file name from Ollama (/api/show). Ask Claude.")
    items_sha = __import__("hashlib").sha256((set_dir / "items.csv").read_bytes()).hexdigest()

    meta_path = out / "run_meta.json"
    fixed = {"model": args.model, "weights_sha256": weights, "question": QUESTION, "options": OPTIONS,
             "items_csv_sha256": items_sha}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if "weights_sha256" not in old:  # folder made by an older version of this script: record weights now
            print(f"note: {meta_path} has no weights fingerprint yet; recording {weights[:12]} "
                  f"(manifest digest was {old.get('model_digest', '?')[:12]}, now {digest[:12]})")
            old["weights_sha256"] = weights
        for k, v in fixed.items():
            if old.get(k) != v:
                sys.exit(f"{out} was started with a different {k}. Use another --out (or --device name).")
        old.setdefault("manifest_digests_seen", [])
        if digest not in old["manifest_digests_seen"]:
            old["manifest_digests_seen"].append(digest)
        meta = old
    else:
        meta = dict(fixed, device=args.device, created_utc=datetime.now(timezone.utc).isoformat(),
                    ollama_version=version, platform=platform.platform(), python=platform.python_version(),
                    model_digest=digest, manifest_digests_seen=[digest])

    # ---- warm-up (loads the model; not timed) and thinking switch
    first = items[0]
    img0 = base64.b64encode((set_dir / "crops" / f"{first['item_id']}.jpg").read_bytes()).decode()
    think_flag = False
    try:
        r = ask(args.host, args.model, img0, first["prompt"], think_flag)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if "think" in body.lower():
            think_flag = None  # model does not support the switch: send nothing
            r = ask(args.host, args.model, img0, first["prompt"], think_flag)
        else:
            sys.exit(f"Ollama error: {e} {body}")
    ps = call(args.host, "/api/ps").get("models", [])
    loaded = next((m for m in ps if m.get("name") == args.model or m.get("model") == args.model), {})
    size, vram = loaded.get("size"), loaded.get("size_vram")
    gpu_share = round(vram / size, 3) if size and vram is not None else None
    meta.setdefault("runs", []).append({"part": args.part, "started_utc": datetime.now(timezone.utc).isoformat(),
                                        "ollama_version": version, "think_flag": think_flag,
                                        "loaded_size_mb": round(size / 2**20) if size else None,
                                        "loaded_on_gpu_mb": round(vram / 2**20) if vram is not None else None,
                                        "share_on_gpu": gpu_share})
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"Ollama {version} | {args.model} weights {weights[:12]} (manifest {digest[:12]}) | model in memory "
          f"{meta['runs'][-1]['loaded_size_mb']} MB, on GPU {meta['runs'][-1]['loaded_on_gpu_mb']} MB "
          f"({'?' if gpu_share is None else f'{100 * gpu_share:.0f}%'}) | thinking switch: {think_flag}")
    msg0 = r.get("message", {})
    print(f"warm-up reply: {msg0.get('content', '')!r}  (done_reason {r.get('done_reason')}, "
          f"hidden thinking {len(msg0.get('thinking') or '')} chars)")
    if not msg0.get("content", "").strip():
        sys.exit("The model gave an EMPTY reply (it probably spent its tokens on hidden 'thinking'). "
                 "Use an instruct tag, e.g. qwen3-vl:4b-instruct. Nothing was recorded.")

    # ---- main loop (resumable)
    ans_path = out / "answers.csv"
    done = set()
    if ans_path.exists():
        done = {row["item_id"] for row in csv.DictReader(open(ans_path, encoding="utf-8"))}
    todo = [r for r in items if r["item_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(items)} items in part '{args.part}', {len(items) - len(todo)} already answered, {len(todo)} to go")
    new_file = not ans_path.exists()
    t_start = time.time()
    counts = {"yes": 0, "no": 0, "unclear": 0}
    unclear_streak = 0
    with open(ans_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, it in enumerate(todo, 1):
            img = base64.b64encode((set_dir / "crops" / f"{it['item_id']}.jpg").read_bytes()).decode()
            t0 = time.perf_counter()
            r = ask_with_retry(args.host, args.model, img, it["prompt"], think_flag)
            wall = time.perf_counter() - t0
            raw = r.get("message", {}).get("content", "")
            a = parse(raw)
            counts[a] += 1
            unclear_streak = unclear_streak + 1 if a == "unclear" else 0
            ns = lambda k: round(r.get(k, 0) / 1e9, 4)
            w.writerow({"item_id": it["item_id"], "set": it["set"], "answer": a, "raw": raw.replace("\n", " ")[:200],
                        "done_reason": r.get("done_reason"),
                        "thinking_chars": len(r.get("message", {}).get("thinking") or ""),
                        "wall_s": round(wall, 4), "total_s": ns("total_duration"), "load_s": ns("load_duration"),
                        "prompt_eval_s": ns("prompt_eval_duration"), "eval_s": ns("eval_duration"),
                        "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count"),
                        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            f.flush()
            if unclear_streak >= 20:
                sys.exit("20 'unclear' replies in a row: the model is not answering yes/no any more. Stopped so the run "
                         "is not filled with junk. Check `ollama ps`, then delete those last rows or ask Claude.")
            if args.part == "pilot" or i % 50 == 0 or i == len(todo):
                el = time.time() - t_start
                print(f"[{i}/{len(todo)}] {it['item_id']:24s} '{it['prompt']}' -> {a:7s} raw={raw.strip()[:30]!r} "
                      f"{wall:.2f}s | avg {el / i:.2f}s/item, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    print(f"\nDone. answers: yes {counts['yes']}, no {counts['no']}, unclear {counts['unclear']}  ->  {ans_path}")


if __name__ == "__main__":
    main()
