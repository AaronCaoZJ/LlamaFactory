#!/usr/bin/env python3
"""
download_0716.py
================
Incrementally fetch the images of the 20260716 cover-hit delivery into `img_0716/`.

Source  : pornpics_tag_cover_dataset_20260716/..._first_batch_hits_20260716.parquet
          (1,740,713 rows, one verified R2 URL each, 4 hosts)
Target  : img_0716/{post_id}_{image_name}      e.g. img_0716/10000050_2.webp
Size    : ~250 GB at the sampled 144 KB/img average.

`img_0716/` is deliberately NOT `img/`: the 1.23M files in `img/` are the `/cut/`
variant (bottom watermark banner stripped) that the existing 110w jsonl points at,
and 135,747 of this delivery's keys collide with them. Writing there would rewrite
the pixels under the old dataset. Kept apart, every image here is exactly what its
parquet URL serves, and the old dataset is untouched.

Resume: writes are atomic (tmp + os.replace), so a file that exists is complete --
existence alone is a sound skip. Safe to kill and rerun; it picks up where it left off.
`--verify` additionally decodes every existing file (slow; for paranoia after a crash).

    python download_0716.py --dry-run     # report what would be fetched, no network
    python download_0716.py               # fetch (resumable)
    MIKOMIKO_CONCURRENCY=64 python download_0716.py
"""
import argparse
import io
import os
import queue
import sys
import threading
import time
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "pornpics_tag_cover_dataset_20260716",
                       "pornpics_category_tag_cover_first_batch_hits_20260716.parquet")
IMG_DIR = os.path.join(HERE, "img_0716")
FAIL_LOG = os.path.join(HERE, "download_0716_failures.tsv")

CONCURRENCY = int(os.environ.get("MIKOMIKO_CONCURRENCY", "32"))
TIMEOUT = 30
RETRIES = 2
UA = {"User-Agent": "Mozilla/5.0 (dataset-probe; research)"}


def load_targets():
    """[(key, url)] from the parquet, key = '{post_id}_{image_name}'.

    The key flattens a URL path that is NOT always /{post_id}/{name} (one host nests a
    collection segment, e.g. /shemale/10108519/2.webp). Flattening is only safe while the
    keys stay unique, so that is asserted rather than assumed -- a silent collision would
    mean two different images overwriting one filename.
    """
    f = pq.ParquetFile(PARQUET)
    out = []
    for b in f.iter_batches(batch_size=200_000, columns=["post_id", "image_name", "url"]):
        d = b.to_pydict()
        out.extend((f"{pid}_{img}", url)
                   for pid, img, url in zip(d["post_id"], d["image_name"], d["url"]))
    keys = {k for k, _ in out}
    if len(keys) != len(out):
        sys.exit(f"[fatal] key collision: {len(out)} rows -> {len(keys)} unique "
                 f"'{{post_id}}_{{image_name}}' keys. Flat naming would silently drop images.")
    return out


def existing(verify):
    """Names already on disk. Atomic writes make existence sufficient; --verify decodes too."""
    if not os.path.isdir(IMG_DIR):
        return set()
    names = {e.name for e in os.scandir(IMG_DIR) if e.is_file() and not e.name.endswith(".tmp")}
    if not verify:
        return names
    ok = set()
    for i, n in enumerate(sorted(names), 1):
        try:
            Image.open(os.path.join(IMG_DIR, n)).verify()
            ok.add(n)
        except Exception:
            pass
        if i % 50_000 == 0:
            print(f"  [verify] {i}/{len(names)} decoded, {len(ok)} good", flush=True)
    print(f"[verify] {len(ok)}/{len(names)} decode OK; {len(names) - len(ok)} will be refetched",
          flush=True)
    return ok


def fetch(url):
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def worker(q, ctr, lock, fails, t0, total):
    while True:
        try:
            key, url = q.get_nowait()
        except queue.Empty:
            return
        data = fetch(url)
        ok = False
        if data:
            try:
                Image.open(io.BytesIO(data)).verify()   # never write bytes that don't decode
                dst = os.path.join(IMG_DIR, key)
                tmp = f"{dst}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(tmp, "wb") as fp:
                    fp.write(data)
                os.replace(tmp, dst)                    # atomic -> present implies complete
                ok = True
            except Exception:
                pass
        with lock:
            ctr["done"] += 1
            if ok:
                ctr["ok"] += 1
                ctr["bytes"] += len(data)
            else:
                fails.append((key, url))
            if ctr["done"] % 2000 == 0:
                el = time.time() - t0
                rate = ctr["done"] / el
                eta = (total - ctr["done"]) / rate / 3600 if rate else 0
                gb = ctr["bytes"] / 1e9
                print(f"  [dl] {ctr['done']:,}/{total:,} ok={ctr['ok']:,} "
                      f"fail={len(fails):,} {rate:.0f}img/s {gb:.1f}GB ETA={eta:.1f}h", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Fetch 20260716 cover-hit images -> img_0716/")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, touch no network")
    ap.add_argument("--verify", action="store_true", help="decode every existing file (slow)")
    ap.add_argument("--limit", type=int, default=None, help="cap fetches (smoke test)")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    t0 = time.time()
    targets = load_targets()
    have = existing(args.verify)
    todo = [(k, u) for k, u in targets if k not in have]
    print(f"[plan] parquet={len(targets):,}  on disk={len(have):,}  to fetch={len(todo):,}"
          f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[plan] dir={IMG_DIR}  concurrency={CONCURRENCY}", flush=True)
    if args.limit:
        todo = todo[:args.limit]
        print(f"[plan] --limit -> fetching {len(todo):,}", flush=True)
    if args.dry_run:
        print("[plan] --dry-run, stopping before any network I/O.", flush=True)
        return
    if not todo:
        print("[dl] nothing to do; img_0716/ already complete.", flush=True)
        return

    q = queue.Queue()
    for t in todo:
        q.put(t)
    ctr = {"done": 0, "ok": 0, "bytes": 0}
    fails, lock, t1 = [], threading.Lock(), time.time()
    threads = [threading.Thread(target=worker, args=(q, ctr, lock, fails, t1, len(todo)),
                               daemon=True) for _ in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Rewritten every run, and removed when a run has no failures: a stale list of
    # already-refetched keys reads exactly like a live one.
    if fails:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            for k, u in fails:
                f.write(f"{k}\t{u}\n")
    elif os.path.exists(FAIL_LOG):
        os.remove(FAIL_LOG)
    el = time.time() - t1
    print(f"[dl] DONE ok={ctr['ok']:,} fail={len(fails):,} of {len(todo):,} "
          f"in {el / 3600:.2f}h ({ctr['bytes'] / 1e9:.1f}GB)", flush=True)
    if fails:
        print(f"[dl] {len(fails):,} failures -> {FAIL_LOG}. Rerunning retries only those.",
              flush=True)


if __name__ == "__main__":
    main()
