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

`nanochat/export.py` exports with `torch.onnx.export(..., dynamo=True)` at opset 18, then checks the
result against these rules and against the model's own logits in onnxruntime. A model that fails
crashes the round, and the reason is in the round's log. The root's script also exports the model
once before training, so that a change that can't be exported fails in seconds.

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
