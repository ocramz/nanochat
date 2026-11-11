"""
check ROUNDS BASELINE_OUT CANDIDATE_OUT SECRET --wall-secs L --rounds N: the speedrun's gate.

Locked, and run as the scorer once per attempt, after its rounds (design §8.2 of
pi-autoresearch-serverless). It passes, exiting 0, when every candidate round:
- took at most L seconds of wall time, from $ROUNDS (baseline rounds are the root's, and not gated:
  a failed gate is final, and would end a search's A/A attempts);
- left one output, model.onnx, which keeps the contract (nanochat/onnx_rules.py);
- is no larger than SIZE_CAP times the same round's baseline model, and evaluates in no more than
  TIME_CAP times its time, so that no training moves into the untimed evaluation;
- is causal: with the held-out text spliced after a random position p, its logits up to p don't
  change. PROBES positions are tried over the attempt, spread over its rounds.
It says why it fails on stderr, which goes to the owner's check.log alone. It never runs the
candidate's code: models run in onnxruntime, as data.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the locked eval/, root's
import evallib  # noqa: E402

rules = evallib.rules

# Positions probed for causality over an attempt. A model that looks ahead at a fraction q of
# positions passes with probability (1 - q)^PROBES: at q = 1%, 4.5e-4.
PROBES = 768
# How far a prefix's logits may move when the text after it changes: the root's don't move at all.
TOLERANCE = 1e-4
# The forwards timed for the evaluation's cap, after the session's creation.
TIMED_FORWARDS = 16


class Failed(Exception):
    pass


def budget(entries, wall_secs, n_rounds):
    """Every candidate round is in $ROUNDS, and took at most wall_secs."""
    candidate = {e["round"]: e for e in entries if e["role"] == "candidate"}
    if sorted(candidate) != list(range(n_rounds)):
        raise Failed(f"candidate rounds {sorted(candidate)}, not 0..{n_rounds - 1}")
    for r, e in sorted(candidate.items()):
        if e["wall_ms"] > wall_secs * 1000:
            raise Failed(f"round {r}: the candidate took {e['wall_ms'] / 1000:.1f} s, over the budget of {wall_secs:g} s")


def timed_eval(data, rows):
    """Seconds to make a session on `data` and run TIMED_FORWARDS rows through it, and the session."""
    t0 = time.perf_counter()
    sess = evallib.session(data)
    for row in rows[:TIMED_FORWARDS]:
        evallib.logits(sess, row[:-1])
    return time.perf_counter() - t0, sess


def causal(sess, rows, tokens, probes, rng):
    """Raises Failed if, at any of `probes` random positions p of random rows, the logits up to p
    change when the tokens after p are replaced by others of the held-out text."""
    t = rules.T_EVAL
    for _ in range(probes):
        x = rows[rng.integers(len(rows))][:-1]
        p = int(rng.integers(t - 1))  # the last position kept; t - 1 - p tokens follow it
        start = int(rng.integers(len(tokens) - t))
        spliced = np.concatenate([x[: p + 1], tokens[start : start + t - 1 - p]])
        before = evallib.logits(sess, x)[: p + 1]
        after = evallib.logits(sess, spliced)[: p + 1]
        diff = float(np.abs(before - after).max())
        if diff > TOLERANCE:
            raise Failed(f"the logits up to position {p} change by {diff:.3g} when the text after it does: the model looks ahead")


def check_round(candidate, baseline, rows, tokens, probes, rng):
    """Raises Failed unless the candidate's model (bytes) keeps the contract, the caps against the
    baseline's model of the same round, and causality. Returns a line on what it measured."""
    if len(candidate) > rules.SIZE_CAP * len(baseline):
        raise Failed(f"the model is {len(candidate)} bytes, over {rules.SIZE_CAP} times the root's {len(baseline)}")
    try:
        t_base, _ = timed_eval(baseline, rows)
        t_cand, sess = timed_eval(candidate, rows)
        if t_cand > rules.TIME_CAP * t_base:
            raise Failed(f"the model evaluates in {t_cand:.2f} s, over {rules.TIME_CAP} times the root's {t_base:.2f} s")
        causal(sess, rows, tokens, probes, rng)
    except evallib.Refused as e:
        raise Failed(str(e)) from e
    return f"{len(candidate)} bytes, eval {t_cand:.2f} s against {t_base:.2f} s, {probes} probes"


def main(argv):
    if len(argv) != 8 or argv[4] != "--wall-secs" or argv[6] != "--rounds":
        print("usage: check ROUNDS BASELINE_OUT CANDIDATE_OUT SECRET --wall-secs L --rounds N", file=sys.stderr)
        return 2
    rounds_json, baseline_out, candidate_out, secret = argv[:4]
    wall_secs, n_rounds = float(argv[5]), int(argv[7])
    try:
        with open(rounds_json) as f:
            budget(json.load(f), wall_secs, n_rounds)
        enc, bos = evallib.tokenizer()
        tokens = evallib.stream(enc, bos, evallib.documents(secret, "val"))
        rows = evallib.chunks(tokens, rules.T_EVAL)
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        per_round = -(-PROBES // n_rounds)
        for r in range(n_rounds):
            try:
                candidate = evallib.read_model(os.path.join(candidate_out, str(r)))
                baseline = evallib.read_model(os.path.join(baseline_out, str(r)))
                said = check_round(candidate, baseline, rows, tokens, per_round, rng)
            except (evallib.Refused, OSError, Failed) as e:
                raise Failed(f"round {r}: {e}") from e
            print(f"check: round {r} passes: {said}", file=sys.stderr)
    except Failed as e:
        print(f"check: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
