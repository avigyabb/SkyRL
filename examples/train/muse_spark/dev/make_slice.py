"""Build an N-layer slice of Muse-Spark-Open-816B-A42B by HTTP range-reading only the needed tensors.

Keeps the embeddings, final norm, lm_head, the first N decoder layers, and the full vision/audio
towers (so the slice still loads as ``MuseSparkForConditionalGeneration``). The per-layer schedules
in config.json are truncated to N.

Usage: HF_TOKEN=... python make_slice.py <out_dir> [num_layers]
"""

import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import subprocess
import time

REPO = "meta-models/Muse-Spark-Open-816B-A42B"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
TOKEN = os.environ.get("HF_TOKEN", "")
SHARD_BYTES = 4 << 30


LOCAL = os.environ.get("MUSE_LOCAL_DIR")


def get(name, start=None, end=None):
    if LOCAL and os.path.exists(os.path.join(LOCAL, name)):
        with open(os.path.join(LOCAL, name), "rb") as f:
            if start is None:
                return f.read()
            f.seek(start)
            return f.read(end - start + 1)
    # curl, not urllib: urllib loses the Range header across HF's CDN redirect and streams the
    # whole shard.
    # HTTP/1.1: HTTP/2 streams from this box get reset mid-transfer (curl exit 92).
    cmd = ["curl", "-sfL", "--http1.1", "--retry", "5", "--retry-all-errors", "-H", f"Authorization: Bearer {TOKEN}"]
    if start is not None:
        cmd += ["-r", f"{start}-{end}"]
    for attempt in range(10):
        proc = subprocess.run(cmd + [BASE + name], capture_output=True)
        data = proc.stdout
        if proc.returncode == 0 and (start is None or len(data) == end - start + 1):
            return data
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{name} [{start}-{end}]: curl exit {proc.returncode}, got {len(data)} bytes")


def keep(key, n):
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
    if m:
        return int(m.group(1)) < n
    return True  # embeddings, norm, lm_head, vision/audio towers, projector


def main():
    out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(out, exist_ok=True)
    index = json.loads(get("model.safetensors.index.json"))["weight_map"]
    wanted = sorted(k for k in index if keep(k, n))

    headers = {}
    for shard in sorted({index[k] for k in wanted}):
        size = struct.unpack("<Q", get(shard, 0, 7))[0]
        headers[shard] = (8 + size, json.loads(get(shard, 8, 8 + size - 1)))

    def fetch(key):
        shard = index[key]
        data_start, header = headers[shard]
        meta = header[key]
        lo, hi = meta["data_offsets"]
        return key, meta, get(shard, data_start + lo, data_start + hi - 1)

    # Pack into ~4GB output shards, written as fetched.
    groups, cur, cur_bytes = [], [], 0
    for key in wanted:
        lo, hi = headers[index[key]][1][key]["data_offsets"]
        if cur and cur_bytes + (hi - lo) > SHARD_BYTES:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(key)
        cur_bytes += hi - lo
    groups.append(cur)

    new_index = {}
    total = len(groups)
    with cf.ThreadPoolExecutor(32) as pool:
        for i, group in enumerate(groups):
            fname = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            for key in group:
                new_index[key] = fname
            if os.path.exists(os.path.join(out, fname)):
                continue
            results = sorted(pool.map(fetch, group))
            header, offset, blobs = {}, 0, []
            for key, meta, blob in results:
                header[key] = {
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "data_offsets": [offset, offset + len(blob)],
                }
                offset += len(blob)
                blobs.append(blob)
                new_index[key] = fname
            hbytes = json.dumps(header, separators=(",", ":")).encode()
            hbytes += b" " * (-len(hbytes) % 8)
            tmp = os.path.join(out, fname + ".tmp")
            with open(tmp, "wb") as f:
                f.write(struct.pack("<Q", len(hbytes)))
                f.write(hbytes)
                for blob in blobs:
                    f.write(blob)
            os.replace(tmp, os.path.join(out, fname))
            print(f"wrote {fname} ({offset / 1e9:.1f} GB, {len(group)} tensors)", flush=True)

    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_index}, f, indent=2)

    config = json.loads(get("config.json"))
    text = config["text_config"]
    text["num_hidden_layers"] = n
    for field in ("layer_types", "layer_rope_theta", "mlp_layer_types"):
        text[field] = text[field][:n]
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "processor_config.json",
        "LICENSE.md",
        "USE_POLICY.md",
    ):
        with open(os.path.join(out, name), "wb") as f:
            f.write(get(name))
    print("done", out)


if __name__ == "__main__":
    main()
