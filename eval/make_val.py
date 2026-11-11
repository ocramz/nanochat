"""
make_val VAL_SHARD TRAIN_SHARD... : writes the held-out splits, eval/secret/val.jsonl.gz and
eval/secret/test.jsonl.gz, and their manifest, eval/val_manifest.json. Run once, on the workstation.

Each split is documents of ClimbMix's last shard (shard_06542, which the image never fetches), in
the shard's order, each at least EVAL_TOKENS tokens in all, none of them also in a training shard
(by the SHA-256 of its text), the test split after the val split. One document per line, as a JSON
string; gzip with no name and no time, so that the files are the same each time they're made.
"""

import gzip
import hashlib
import io
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evallib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REVISION = "915333b4f8b8684f39aeaafea600fea6f43fb703"  # karpathy/climbmix-400b-shuffle


def texts(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        yield from pf.read_row_group(rg).column("text").to_pylist()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, docs):
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
        for doc in docs:
            z.write((json.dumps(doc, ensure_ascii=False) + "\n").encode("utf-8"))
    with open(path, "wb") as f:
        f.write(raw.getvalue())


def main(val_shard, train_shards):
    enc, _ = evallib.tokenizer()
    seen = {digest(t) for shard in train_shards for t in texts(shard)}
    splits = {"val": [], "test": []}
    counts = {"val": 0, "test": 0}
    dropped = 0
    order = ["val", "test"]
    for text in texts(val_shard):
        if not order:
            break
        if digest(text) in seen:
            dropped += 1
            continue
        split = order[0]
        splits[split].append(text)
        counts[split] += 1 + len(enc.encode_ordinary(text))
        if counts[split] >= evallib.EVAL_TOKENS:
            order.pop(0)
    if order:
        raise SystemExit(f"the shard has too few documents for {order}")
    os.makedirs(os.path.join(HERE, "secret"), exist_ok=True)
    manifest = {
        "dataset": "karpathy/climbmix-400b-shuffle",
        "revision": REVISION,
        "val_shard": {"file": os.path.basename(val_shard), "sha256": sha256_file(val_shard)},
        "train_shards": [{"file": os.path.basename(s), "sha256": sha256_file(s)} for s in train_shards],
        "dropped_as_in_training": dropped,
        "eval_tokens": evallib.EVAL_TOKENS,
    }
    for split in ("val", "test"):
        path = os.path.join(HERE, "secret", f"{split}.jsonl.gz")
        write(path, splits[split])
        manifest[split] = {"documents": len(splits[split]), "tokens": counts[split], "sha256": sha256_file(path)}
    with open(os.path.join(HERE, "val_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_val VAL_SHARD TRAIN_SHARD...")
    main(sys.argv[1], sys.argv[2:])
