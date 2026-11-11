"""
What eval/check.py, eval/score.py and eval/diagnose.py share: the tokenizer, the held-out text as one
stream of tokens, onnxruntime sessions, bits per byte, and a round's snapshots.

Locked, and run as the scorer under `python -I`: it imports numpy, onnx, onnxruntime and tiktoken
from the image, and the contract (nanochat/onnx_rules.py) by its path in the locked tree, never the
nanochat package, whose other files agents change. Nothing is unpickled. A model is handed to
onnxruntime as bytes, never as a path, once it keeps the contract, which allows no external data.
"""

import base64
import gzip
import importlib.util
import json
import math
import os
import re

import numpy as np

# onnxruntime sends telemetry to Microsoft from a thread of its own on Linux, which the sandbox denies
# (no network sockets): off before it loads. The commands set it too.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import tiktoken  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(TREE, "bench", "tokenizer")
# Each split's path below its root: $SECRET for val and test, $DIAGNOSTIC for the diagnostic text,
# which nothing selects on (plan 16e).
SPLITS = {
    "val": "eval/secret/val.jsonl.gz",
    "test": "eval/secret/test.jsonl.gz",
    "diagnostic": "eval/diagnostic/diagnostic.jsonl.gz",
}
# The tokens of a split that are scored: the first ones of its stream.
EVAL_TOKENS = 2**18
# The diagnostic text's tokens, which the final model's breakdown scores, and the first ones of a
# stream that a curve's every point scores (plan 16e).
DIAGNOSTIC_TOKENS = 2**16
CURVE_TOKENS = 2**15
# The snapshots of a round that count, at most, and how much larger than the final model one may be.
MAX_SNAPSHOTS = 16
SNAPSHOT_SLACK = 1.1
SNAPSHOT_NAME = re.compile(r"^(0|[1-9][0-9]{0,11})\.onnx$")


def _load_rules():
    spec = importlib.util.spec_from_file_location("onnx_rules", os.path.join(TREE, "nanochat", "onnx_rules.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rules = _load_rules()


class Refused(Exception):
    """A model the contract or the gate refuses: the message says why."""


def tokenizer():
    """The tokenizer's encoding (bench/tokenizer), and its BOS id."""
    with open(os.path.join(TOKENIZER, "tokenizer.json")) as f:
        meta = json.load(f)
    ranks = {}
    with open(os.path.join(TOKENIZER, "ranks.tiktoken"), "rb") as f:
        for line in f:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    enc = tiktoken.Encoding(
        name="speedrun", pat_str=meta["pat_str"], mergeable_ranks=ranks, special_tokens=meta["special_tokens"]
    )
    return enc, meta["special_tokens"][meta["bos"]]


def token_bytes(enc):
    """Each token's length in bytes, 0 for the special tokens, which bpb doesn't count (nanochat's
    definition: the raw bytes, never decoded to a string first)."""
    specials = {enc.encode_single_token(name) for name in enc.special_tokens_set}
    return np.array(
        [0 if i in specials else len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)],
        dtype=np.int64,
    )


def documents(root, split):
    """A split's documents, below its root: $SECRET, or $DIAGNOSTIC for the diagnostic text."""
    with gzip.open(os.path.join(root, SPLITS[split]), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stream(enc, bos, docs, n_tokens=EVAL_TOKENS):
    """The first `n_tokens` of the documents' tokens, each document after a BOS."""
    out = []
    total = 0
    for ids in enc.encode_ordinary_batch(docs, num_threads=4):
        out.append(bos)
        out.extend(ids)
        total += 1 + len(ids)
        if total >= n_tokens:
            break
    if total < n_tokens:
        raise RuntimeError(f"the split has {total} tokens, fewer than {n_tokens}")
    return np.array(out[:n_tokens], dtype=np.int64)


def chunks(tokens, t_eval):
    """Non-overlapping rows of t_eval + 1 tokens: inputs [:-1], targets [1:]."""
    n = len(tokens) // (t_eval + 1)
    return tokens[: n * (t_eval + 1)].reshape(n, t_eval + 1)


def threads():
    """The CPUs the container's cgroup allows (cgroup2, or v1 as on Cloud Run), or the machine's."""
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            with open(quota_file) as f:
                words = f.read().split()
            if period_file is not None:
                with open(period_file) as f:
                    words.append(f.read().strip())
            quota, period = words[0], words[-1]
            if quota not in ("max", "-1"):
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError, IndexError):
            pass
    return os.cpu_count() or 1


def read_model(directory):
    """The bytes of the round's model, which must be the round's only output."""
    names = sorted(os.listdir(directory))
    if names != ["model.onnx"]:
        raise Refused(f"the round's outputs are {names}, not ['model.onnx'] alone")
    with open(os.path.join(directory, "model.onnx"), "rb") as f:
        return f.read()


def session(data):
    """An onnxruntime session on a model that keeps the contract."""
    found = rules.problems(data)
    if found:
        raise Refused("the model breaks the contract:\n  " + "\n  ".join(found))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads()
    options.inter_op_num_threads = 1
    try:
        return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise Refused(f"onnxruntime refuses the model: {type(e).__name__}: {e}") from e


def logits(sess, x):
    """The model's logits for one row x of t_eval tokens, the first `VOCAB` columns."""
    out = sess.run(None, {sess.get_inputs()[0].name: x.reshape(1, -1)})[0]
    out = np.asarray(out, dtype=np.float32)[0, :, : rules.VOCAB]
    if not np.isfinite(out).all():
        raise Refused("the model's logits are not all finite")
    return out


def target_nats(sess, row, nbytes):
    """One row's targets: the nats of each under the model's logits, normalized here by a
    log-softmax (float32 terms, float64 sums), and each one's bytes."""
    x, y = row[:-1], row[1:]
    z = logits(sess, x)
    top = z.max(axis=1, keepdims=True)
    lse = top[:, 0].astype(np.float64) + np.log(np.exp(z - top).sum(axis=1, dtype=np.float64))
    return lse - z[np.arange(len(y)), y].astype(np.float64), nbytes[y]


def bits_per_byte(sess, rows, nbytes):
    """Bits per byte over the rows' targets with bytes: the nats of the targets under the model's
    logits over ln 2 times their bytes."""
    nats = 0.0
    total = 0
    for row in rows:
        nll, b = target_nats(sess, row, nbytes)
        nats += float((nll * (b > 0)).sum())
        total += int(b.sum())
    return nats / (math.log(2) * total)


def nats_by_position(sess, rows, nbytes):
    """For each position of a row's targets, over the rows: the nats of the targets with bytes, and
    their bytes. Bits per byte over positions lo..hi is nats[lo:hi].sum() / (ln 2 · bytes[lo:hi].sum())."""
    nats = np.zeros(rows.shape[1] - 1, dtype=np.float64)
    total = np.zeros(rows.shape[1] - 1, dtype=np.int64)
    for row in rows:
        nll, b = target_nats(sess, row, nbytes)
        nats += nll * (b > 0)
        total += b
    return nats, total


def skeleton(data):
    """A model with its initializers' values dropped: what a snapshot must share with the final
    model, so that scoring it costs what scoring the gated model does."""
    model = onnx.load_model_from_string(data)
    for tensor in model.graph.initializer:
        for field in ("raw_data", "float_data", "int32_data", "int64_data", "double_data", "uint64_data", "string_data"):
            tensor.ClearField(field)
    return model.SerializeToString()


def snapshots(directory, wall_ms, final, say):
    """The snapshots of a round that count (plan 16e), in time order, one at a time, as (ms, bytes):
    named `<ms>.onnx`, ms under the round's `wall_ms`, the first MAX_SNAPSHOTS by time, each a
    regular file at most SNAPSHOT_SLACK times the final model's bytes, keeping the contract, with the
    final model's graph but for its weights. `say` is told why each other one is skipped. A
    directory that isn't there holds none."""
    if directory is None or not os.path.isdir(directory):
        return
    named = []
    for name in sorted(os.listdir(directory)):
        match = SNAPSHOT_NAME.match(name)
        if match is None:
            say(f"{name}: not named <ms>.onnx")
        elif int(match.group(1)) >= wall_ms:
            say(f"{name}: not before the round's end, {wall_ms} ms")
        else:
            named.append((int(match.group(1)), name))
    named.sort()
    if len(named) > MAX_SNAPSHOTS:
        say(f"{len(named)} snapshots: the first {MAX_SNAPSHOTS} count")
        named = named[:MAX_SNAPSHOTS]
    want = None
    for ms, name in named:
        path = os.path.join(directory, name)
        if os.path.islink(path) or not os.path.isfile(path):
            say(f"{name}: not a regular file")
            continue
        if os.path.getsize(path) > SNAPSHOT_SLACK * len(final):
            say(f"{name}: larger than the final model")
            continue
        with open(path, "rb") as f:
            data = f.read()
        found = rules.problems(data)
        if found:
            say(f"{name}: breaks the contract: {found[0]}")
            continue
        want = skeleton(final) if want is None else want
        if skeleton(data) != want:
            say(f"{name}: not the final model's graph")
            continue
        yield ms, data
