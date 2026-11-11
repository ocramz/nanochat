"""
score OUTPUTS SECRET [--split val|test]: one round's val bits per byte, one number on stdout.

Locked, and run as the scorer once per measured round of each side (design §8.3 of
pi-autoresearch-serverless): the round's model, $OUTPUTS/model.onnx, checked against the contract
(nanochat/onnx_rules.py) and run in onnxruntime, never the candidate's code, on the held-out text in
$SECRET, which the payload never sees. The test split is for scoring by hand: no search selects on it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the locked eval/, root's
import evallib  # noqa: E402


def main(argv):
    if len(argv) not in (2, 4) or (len(argv) == 4 and argv[2] != "--split"):
        print("usage: score OUTPUTS SECRET [--split val|test]", file=sys.stderr)
        return 2
    outputs, secret = argv[0], argv[1]
    split = argv[3] if len(argv) == 4 else "val"
    enc, bos = evallib.tokenizer()
    rows = evallib.chunks(evallib.stream(enc, bos, evallib.documents(secret, split)), evallib.rules.T_EVAL)
    try:
        sess = evallib.session(evallib.read_model(outputs))
        bpb = evallib.bits_per_byte(sess, rows, evallib.token_bytes(enc))
    except evallib.Refused as e:
        print(f"score: {e}", file=sys.stderr)
        return 1
    print(f"{bpb:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
