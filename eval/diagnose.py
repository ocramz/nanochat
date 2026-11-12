"""
diagnose CHECKPOINTS OUTPUTS WALL_MS DIAGNOSTIC: one round's learning curve and the final model's
breakdown, on the diagnostic text, as one JSON object on stdout (plan 16e of pi-autoresearch-serverless).

Locked, and run as the scorer once for each side's first measured round, after the measurement
(design §8.3): agents see what it prints, under the names `[diagnose]` gives, so it scores only the
diagnostic text, in $DIAGNOSTIC, which nothing selects on and no round can read; never val, and it
isn't given $SECRET. Its models run in onnxruntime, as data, never the candidate's code.

- The curves: each snapshot of $CHECKPOINTS that counts (evallib.snapshots), then the final model,
  $OUTPUTS/model.onnx, scored once each on the diagnostic text's first CURVE_TOKENS tokens, and drawn
  against two x (plan 16f): `bpb_time`, seconds since the round's start, the snapshot's name, and the
  round's wall time, $WALL_MS, for the final model; `bpb_tokens`, the tokens each model says it trained
  on, in its metadata, leaving out a model that says none, or no more than the point before it.
- The values `bpb_pos_<lo>_<hi>`: the final model's bits per byte on DIAGNOSTIC_TOKENS tokens, over
  the targets at positions lo..hi of their rows: whether the model uses its context.

Why it skips a snapshot, and how long each took, go to stderr, the owner's diagnose.log.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the locked eval/, root's
import evallib  # noqa: E402

# The positions of a row's targets each value covers, [lo, hi).
BUCKETS = ((0, 64), (64, 256), (256, evallib.rules.T_EVAL))


def say(message):
    print(f"diagnose: {message}", file=sys.stderr, flush=True)


def by_tokens(curve, name, tokens, bpb):
    """Adds a model's point to the curve by tokens trained, if it says how many, more than the point
    before it, so that x stays strictly increasing; else says why not."""
    if tokens is None:
        say(f"{name}: no tokens trained in its metadata, so not on bpb_tokens")
    elif curve and tokens <= curve[-1][0]:
        say(f"{name}: {tokens} tokens trained, no more than the point before, so not on bpb_tokens")
    else:
        curve.append([tokens, bpb])


def main(argv):
    if len(argv) != 4 or not argv[2].isdigit():
        print("usage: diagnose CHECKPOINTS OUTPUTS WALL_MS DIAGNOSTIC", file=sys.stderr)
        return 2
    checkpoints, outputs, wall_ms, diagnostic = argv[0], argv[1], int(argv[2]), argv[3]
    t = evallib.rules.T_EVAL
    enc, bos = evallib.tokenizer()
    nbytes = evallib.token_bytes(enc)
    tokens = evallib.stream(enc, bos, evallib.documents(diagnostic, "diagnostic"), evallib.DIAGNOSTIC_TOKENS)
    rows = evallib.chunks(tokens, t)
    curve_rows = evallib.chunks(tokens[: evallib.CURVE_TOKENS], t)
    try:
        final = evallib.read_model(outputs)
        sess = evallib.session(final)
    except evallib.Refused as e:
        say(f"the final model: {e}")
        return 1
    by_time, by_tokens_trained = [], []
    for ms, tokens, data in evallib.snapshots(checkpoints, wall_ms, final, say):
        started = time.monotonic()
        try:
            bpb = evallib.bits_per_byte(evallib.session(data), curve_rows, nbytes)
        except evallib.Refused as e:
            say(f"{ms}.onnx: {e}")
            continue
        by_time.append([ms / 1000, bpb])
        by_tokens(by_tokens_trained, f"{ms}.onnx", tokens, bpb)
        say(f"{ms} ms, {tokens} tokens: {bpb:.6f} ({time.monotonic() - started:.1f} s)")
    started = time.monotonic()
    bpb = evallib.bits_per_byte(sess, curve_rows, nbytes)
    tokens = evallib.examine(final)[1]
    by_time.append([wall_ms / 1000, bpb])
    by_tokens(by_tokens_trained, "model.onnx", tokens, bpb)
    nats, total = evallib.nats_by_position(sess, rows, nbytes)
    values = {
        f"bpb_pos_{lo}_{hi - 1}": float(nats[lo:hi].sum() / (math.log(2) * total[lo:hi].sum())) for lo, hi in BUCKETS
    }
    say(f"the final model, {wall_ms} ms, {tokens} tokens: {bpb:.6f}; by position {values} ({time.monotonic() - started:.1f} s)")
    print(json.dumps({"curves": {"bpb_time": by_time, "bpb_tokens": by_tokens_trained}, "values": values}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
