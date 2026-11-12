# The CPU speedrun

Train nanochat's GPT from scratch, on the CPU, within a fixed wall-clock budget, so that it predicts
held-out text as well as possible. The metric is **val_bpb**, bits per byte on held-out ClimbMix text
that no run ever sees: lower is better.

## A round

Each round runs, from the repository's root,

    python -m scripts.speedrun --wall-secs L --out $OUTPUTS

with `L` from `autoresearch.toml`, on 8 vCPUs and no GPU, with no network, from a fresh process:
nothing survives from one round to the next. The budget is the round's whole wall time, from the
process's start to its exit: Python's start, imports, data loading, training and the export. A round
over `L` seconds fails the experiment. The root's script stops training early enough to export.

The round writes exactly one file, `$OUTPUTS/model.onnx`: the trained model, which the harness scores.

## The model's contract

The harness never runs the training code to score: it runs the exported model in onnxruntime.
`nanochat/onnx_rules.py` checks the rules below, both in the round (by `nanochat/export.py`) and in
the harness.

- **Interface:** one input, int64 token ids, shape `[1, 512]`; one output, float32 logits, shape
  `[1, 512, V]` with `V >= 8192`. Columns past 8192 are ignored. The harness applies log-softmax
  itself, so the logits need not be normalized.
- **Causal:** the logits at position `t` may depend on tokens `0..t` only. The harness replaces the
  tokens after random positions with other text and refuses a model whose earlier logits change.
- **Ops:** only ops of ONNX's default domain, and none that draws random numbers (RandomNormal,
  RandomUniform and their `Like` forms, Multinomial, Bernoulli, Dropout). No local functions, and no
  external data: every weight inside the one file.
- **Size and speed:** at most 512 MiB, at most 8 times the bytes of the root's model, and an
  evaluation at most 8 times as slow as the root model's.

`nanochat/export.py` exports with `torch.onnx.export(..., dynamo=True, optimize=False)` at opset 18,
then checks the result against these rules and against the model's own logits in onnxruntime. A
model that fails crashes the round, and the reason is in the round's log. The root's script also
exports the model once before training, so that a change that can't be exported fails in seconds.

## Snapshots and learning curves

A round may write snapshots of its model to the directory `--checkpoints` names, `$CHECKPOINTS`: the
root's script writes one at each tenth of its training time. They let you see how a run trained, not
only where it ended.

- **Optional:** a round that writes none is scored as any other.
- **Where and how:** `$CHECKPOINTS/<ms>.onnx`, `<ms>` the milliseconds since the process started, a
  whole number with no leading zero, before the round's end.
- **The final model's graph:** a snapshot counts only if it is `model.onnx` but for its weights'
  values and its metadata, no larger, and keeps the contract; at most 16 count, the earliest. The
  root's `nanochat/export.py` makes one in a fraction of a second, by writing the current weights
  into the untrained model's export, which runs no graph optimizer for that reason.
- **Tokens trained:** each snapshot, and `model.onnx`, may give the tokens it trained on as the ONNX
  metadata entry `tokens`, a whole number in decimal. The root's records `step × TOTAL_BATCH`. A
  model that gives none, or no more than the snapshot before it, is left out of the curve by tokens
  only.
- **What you see:** the harness scores the first measured round's snapshots, and its final model, on
  diagnostic text: held-out text from another shard than the training data and the metric's, which
  no round can read and nothing selects on. The query tools show each experiment's curves of bits per
  byte, `bpb_time` against the seconds since the round's start and `bpb_tokens` against the tokens
  trained, and the final model's bits per byte by the position of the target in its row (0–63,
  64–255, 256–511), baseline against candidate. By time is what the budget buys; by tokens separates
  learning per token from throughput, so that a faster candidate's curve reaches further right. They
  are not the metric, and don't count toward a node's value.
- **Their cost:** writing them takes the round's own time, within its budget, the same for every
  round of either side.

## What may change

- `scripts/speedrun.py`: the training run, its hyperparameters and schedules.
- `nanochat/gpt.py`, `nanochat/optim.py`, `nanochat/dataloader.py`, `nanochat/common.py`,
  `nanochat/flash_attention.py`: the model, the optimizer, the data pipeline.
- `nanochat/export.py`: how the model exports, as long as the result keeps the contract.

Everything else is locked: the tokenizer (`nanochat/tokenizer.py`, `bench/tokenizer`, 8192 tokens),
the dataset's reader (`nanochat/dataset.py`), the rules, and the harness (`bench/`, `eval/`).

## The environment

- PyTorch 2.9.1 for the CPU, onnx, onnxscript, onnxruntime, numpy, pyarrow, tiktoken; nothing else
  can be installed.
- Training data: ClimbMix's shards 0 and 1 in `/opt/nanochat/base_data_climbmix`; nanochat's
  dataloader trains on all but the last file, and takes the last as its own validation split. The
  held-out text the harness scores on is from another shard.
- Threads: the CPUs the container's cgroup allows (`/sys/fs/cgroup/cpu.max`).
- The root runs eagerly: `torch.compile` compiles within the budget, at each round's start.
- A round's seed comes from `os.urandom`, so rounds differ; the harness compares each candidate round
  with a root round run on the same machine.
- No network: a round that opens a network socket, even one that fails, is a policy violation, and
  fails the experiment. onnxruntime's telemetry would: `ORT_DISABLE_TELEMETRY=1` is set for every
  round, and `nanochat/export.py` sets it again before onnxruntime loads.
