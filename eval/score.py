"""
score OUTPUTS SECRET [--checkpoints DIR --wall-ms MS] [--split val|test]: one round's val bits per
byte, one number on stdout.

Locked, and run as the scorer once per measured round of each side (design §8.3 of
pi-autoresearch-serverless): the round's model, $OUTPUTS/model.onnx, checked against the contract
(nanochat/onnx_rules.py) and run in onnxruntime, never the candidate's code, on the held-out text in
$SECRET, which the payload never sees. The test split is for scoring by hand: no search selects on it.

With --checkpoints, the round's snapshots that count (evallib.snapshots) and the final model, at
--wall-ms, are scored on the first CURVE_TOKENS val tokens after the number is printed: the val curve,
on stderr, which goes to the owner's score.log alone (plan 16e), each point with its time and the
tokens its model says it trained on (plan 16f). Its snapshots take at most CURVE_SECS, and nothing in
it changes what is printed, or the exit status.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the locked eval/, root's
import evallib  # noqa: E402

# The most the snapshots of the val curve take, so that a slow model can't push the score past its
# timeout: on Cloud Run the root's nine take about 40 s, after the metric's 35 (plan 16e's trial 3).
CURVE_SECS = 45
USAGE = "usage: score OUTPUTS SECRET [--checkpoints DIR --wall-ms MS] [--split val|test]"


def say(message):
    print(f"score: {message}", file=sys.stderr, flush=True)


def options(argv):
    """The positional arguments and the flags, or None when they don't parse."""
    if len(argv) < 2 or len(argv) % 2 != 0:
        return None
    flags = dict(zip(argv[2::2], argv[3::2]))
    if len(flags) * 2 != len(argv) - 2 or not set(flags) <= {"--checkpoints", "--wall-ms", "--split"}:
        return None
    if ("--checkpoints" in flags) != ("--wall-ms" in flags) or not flags.get("--wall-ms", "0").isdigit():
        return None
    if flags.get("--split", "val") not in ("val", "test"):
        return None
    return argv[0], argv[1], flags


def val_curve(checkpoints, wall_ms, final, sess, tokens, nbytes):
    """The val curve, to stderr: each snapshot's bits per byte, for at most CURVE_SECS, then the final
    model's, on the first CURVE_TOKENS tokens."""
    deadline = time.monotonic() + CURVE_SECS
    rows = evallib.chunks(tokens[: evallib.CURVE_TOKENS], evallib.rules.T_EVAL)
    shown = 0
    for ms, trained, data in evallib.snapshots(checkpoints, wall_ms, final, say):
        if time.monotonic() > deadline:
            say(f"the val curve's snapshots stop after {CURVE_SECS} s")
            break
        try:
            say(f"val curve: {ms} ms, {trained} tokens: {evallib.bits_per_byte(evallib.session(data), rows, nbytes):.6f}")
            shown += 1
        except evallib.Refused as e:
            say(f"{ms}.onnx: {e}")
    if shown > 0:
        trained = evallib.examine(final)[1]
        say(f"val curve: {wall_ms} ms, {trained} tokens: {evallib.bits_per_byte(sess, rows, nbytes):.6f} (the final model)")


def main(argv):
    parsed = options(argv)
    if parsed is None:
        print(USAGE, file=sys.stderr)
        return 2
    outputs, secret, flags = parsed
    enc, bos = evallib.tokenizer()
    tokens = evallib.stream(enc, bos, evallib.documents(secret, flags.get("--split", "val")))
    rows = evallib.chunks(tokens, evallib.rules.T_EVAL)
    nbytes = evallib.token_bytes(enc)
    try:
        final = evallib.read_model(outputs)
        sess = evallib.session(final)
        bpb = evallib.bits_per_byte(sess, rows, nbytes)
    except evallib.Refused as e:
        print(f"score: {e}", file=sys.stderr)
        return 1
    print(f"{bpb:.6f}", flush=True)
    if "--checkpoints" in flags:
        try:
            val_curve(flags["--checkpoints"], int(flags["--wall-ms"]), final, sess, tokens, nbytes)
        except Exception as e:  # the curve is the owner's: it never fails the score
            say(f"the val curve failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
