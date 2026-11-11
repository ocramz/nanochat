"""
The speedrun's training script (SPEEDRUN.md): trains nanochat's GPT from scratch on the CPU, within
--wall-secs seconds of wall time from start to exit, and writes the model, as ONNX, to
--out/model.onnx, which the harness scores by its bits per byte on held-out text.

Trimmed from scripts/base_train.py: one process on the CPU, no evaluation, no checkpoints, no
logging service; the schedules follow the time spent rather than a number of steps, as
karpathy/autoresearch's do. With --checkpoints, it writes a snapshot of the model at each tenth of
its training time, which the harness scores for the learning curves (SPEEDRUN.md). Run from the
repository's root:

    NANOCHAT_BASE_DIR=/opt/nanochat python -m scripts.speedrun --wall-secs 120 --out OUT [--checkpoints DIR]

Editable: everything here is the agents' to change, within SPEEDRUN.md's rules.
"""

import time

START = time.time()  # the budget runs from here (and the interpreter's start, a moment before)

import argparse  # noqa: E402
import gc  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
import torch._dynamo  # noqa: E402

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit  # noqa: E402
from nanochat.export import export, snapshot, template, to_onnx, verify  # noqa: E402
from nanochat.gpt import GPT, GPTConfig  # noqa: E402
from nanochat.onnx_rules import T_EVAL  # noqa: E402
from nanochat.tokenizer import get_tokenizer  # noqa: E402

# -----------------------------------------------------------------------------
# Hyperparameters: nanochat's, at the scale of a few CPU minutes

DEPTH = 4  # transformer layers
ASPECT_RATIO = 64  # model_dim = DEPTH * ASPECT_RATIO, rounded up to HEAD_DIM
HEAD_DIM = 64
SEQ_LEN = T_EVAL  # tokens per row; the exported model reads T_EVAL at once
WINDOW_PATTERN = "L"  # full context: PyTorch's SDPA has no fast sliding window on the CPU
DEVICE_BATCH = 8  # rows per forward/backward
TOTAL_BATCH = DEVICE_BATCH * SEQ_LEN  # tokens per optimizer step
# base_train's learning rates, scaled as it scales them for a batch other than its reference 2^19
B_REF = 2**19
BATCH_LR_SCALE = (TOTAL_BATCH / B_REF) ** 0.5
EMBEDDING_LR = 0.3 * BATCH_LR_SCALE
UNEMBEDDING_LR = 0.008 * BATCH_LR_SCALE
MATRIX_LR = 0.02 * BATCH_LR_SCALE
SCALAR_LR = 0.5 * BATCH_LR_SCALE
WEIGHT_DECAY = 0.28 * BATCH_LR_SCALE  # Muon's cautious weight decay, decayed to 0 over the run
WARMUP_FRAC = 0.05  # of the training time, linear
WARMDOWN_FRAC = 0.65  # of the training time, linear down to FINAL_LR_FRAC
FINAL_LR_FRAC = 0.05
MOMENTUM_RAMP_STEPS = 300  # Muon's momentum, 0.85 to 0.95
# Seconds kept, out of the budget, for the export after training and the exit.
RESERVE_SECS = 15.0
# torch.compile: off, since compiling takes longer than it saves in a run this short. The model and
# the optimizer's fused steps then run eagerly.
COMPILE = False
# Snapshots for the learning curves, with --checkpoints: one at each 1/SNAPSHOTS of the training time
# but its end, which the exported model stands for.
SNAPSHOTS = 10


def cpu_threads():
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


def say(message):
    print(f"[{time.time() - START:7.1f}s] {message}", file=sys.stderr, flush=True)


def lr_multiplier(progress):
    if progress < WARMUP_FRAC:
        return (progress + 1e-3) / WARMUP_FRAC
    if progress <= 1.0 - WARMDOWN_FRAC:
        return 1.0
    down = (1.0 - progress) / WARMDOWN_FRAC
    return down * 1.0 + (1.0 - down) * FINAL_LR_FRAC


def muon_momentum(step):
    frac = min(step / MOMENTUM_RAMP_STEPS, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def main():
    parser = argparse.ArgumentParser(description="the speedrun's training run")
    parser.add_argument("--wall-secs", type=float, required=True, help="the budget, start to exit")
    parser.add_argument("--out", required=True, help="the directory model.onnx goes in")
    parser.add_argument("--checkpoints", help="the directory snapshots go in, as <ms since the start>.onnx")
    args = parser.parse_args()
    deadline = START + args.wall_secs - RESERVE_SECS

    torch._dynamo.config.disable = not COMPILE
    torch.set_num_threads(cpu_threads())
    seed = int.from_bytes(os.urandom(8), "little") % 2**63
    torch.manual_seed(seed)
    say(f"threads {torch.get_num_threads()}, seed {seed}")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model_dim = -(-DEPTH * ASPECT_RATIO // HEAD_DIM) * HEAD_DIM
    config = GPTConfig(
        sequence_len=SEQ_LEN, vocab_size=vocab_size, n_layer=DEPTH, n_head=model_dim // HEAD_DIM,
        n_kv_head=model_dim // HEAD_DIM, n_embd=model_dim, window_pattern=WINDOW_PATTERN,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    say(f"{config}, {sum(p.numel() for p in model.parameters()):,} parameters")

    # Fail fast: a model that won't export crashes the round now, not after its budget. The export is
    # also the template every snapshot fills in.
    untrained = to_onnx(model)
    verify(model, untrained)
    shape = template(untrained) if args.checkpoints else None
    del untrained
    say("the untrained model exports")

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR, matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY, scalar_lr=SCALAR_LR,
    )
    trained = torch.compile(model, dynamic=False) if COMPILE else model
    assert TOTAL_BATCH % (DEVICE_BATCH * SEQ_LEN) == 0
    grad_accum = TOTAL_BATCH // (DEVICE_BATCH * SEQ_LEN)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, DEVICE_BATCH, SEQ_LEN, split="train", device="cpu",
    )
    x, y, _ = next(loader)

    train_start = time.time()
    train_secs = deadline - train_start
    if train_secs <= 0:
        raise SystemExit(f"no time left to train: {train_secs:.1f} s")
    say(f"training for {train_secs:.1f} s")
    step = 0
    smooth_loss = 0.0
    snapped = 0
    while True:
        progress = (time.time() - train_start) / train_secs
        if progress >= 1.0:
            break
        if shape is not None and snapped + 1 < SNAPSHOTS and progress >= (snapped + 1) / SNAPSHOTS:
            snapped += 1
            t0 = time.time()
            ms = int((t0 - START) * 1000)
            data = snapshot(shape, model)
            with open(os.path.join(args.checkpoints, f"{ms}.onnx"), "wb") as f:
                f.write(data)
            say(f"snapshot {snapped} at {ms} ms, {len(data):,} bytes, in {time.time() - t0:.2f} s")
        lrm = lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum(step)
                group["weight_decay"] = WEIGHT_DECAY * 0.5 * (1 + math.cos(math.pi * progress))
        for _ in range(grad_accum):
            loss = trained(x, y)
            (loss / grad_accum).backward()
            x, y, _ = next(loader)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()
        step += 1
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        if step % 25 == 0:
            elapsed = time.time() - train_start
            say(f"step {step} | {100 * progress:.0f}% | loss {smooth_loss / (1 - 0.9**step):.4f} | lrm {lrm:.2f} | {step * TOTAL_BATCH / elapsed:,.0f} tok/s")

    elapsed = time.time() - train_start
    say(f"trained {step} steps, {step * TOTAL_BATCH:,} tokens in {elapsed:.1f} s, loss {smooth_loss / (1 - 0.9**max(step, 1)):.4f}")
    size = export(model, os.path.join(args.out, "model.onnx"))
    say(f"exported {size:,} bytes")


if __name__ == "__main__":
    main()
