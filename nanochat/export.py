"""
Exports the trained model as ONNX, the speedrun's only output (SPEEDRUN.md), and checks it as the
gate will: the contract in nanochat/onnx_rules.py (locked), then onnxruntime against the model's own
logits. A model that fails raises ExportFailed, so the round crashes with the reason in its log, which
agents read; the gate's own reasons never reach them.

Editable: change how the model exports, as long as what it writes keeps the contract.
"""

import logging

import numpy as np
import onnxruntime as ort
import torch

from nanochat.onnx_rules import T_EVAL, VOCAB, problems

for _name in ("onnxscript", "onnx_ir", "torch.onnx"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class ExportFailed(Exception):
    pass


def to_onnx(model):
    """The model's bytes as ONNX: one int64 input [1, T_EVAL], one float output [1, T_EVAL, VOCAB].
    Export the plain module, never a torch.compile'd one."""
    was_training = model.training
    model.eval()
    idx = torch.zeros((1, T_EVAL), dtype=torch.long)
    try:
        with torch.no_grad():
            program = torch.onnx.export(
                model, (idx,), dynamo=True, opset_version=18,
                input_names=["idx"], output_names=["logits"], verbose=False,
            )
        program.optimize()
        return program.model_proto.SerializeToString()
    except Exception as e:
        raise ExportFailed(f"torch.onnx.export failed: {type(e).__name__}: {e}") from e
    finally:
        model.train(was_training)


def verify(model, data, tolerance=1e-3):
    """Raises ExportFailed unless `data` keeps the contract, loads in onnxruntime, and gives the
    model's own logits on random tokens."""
    found = problems(data)
    if found:
        raise ExportFailed("the exported model breaks the contract (SPEEDRUN.md):\n  " + "\n  ".join(found))
    try:
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise ExportFailed(f"onnxruntime refuses the exported model: {type(e).__name__}: {e}") from e
    was_training = model.training
    model.eval()
    idx = torch.randint(0, VOCAB, (1, T_EVAL))
    with torch.no_grad():
        want = model(idx).float().numpy()[..., :VOCAB]
    model.train(was_training)
    got = session.run(None, {session.get_inputs()[0].name: idx.numpy()})[0][..., :VOCAB]
    if not np.isfinite(got).all():
        raise ExportFailed("onnxruntime's logits are not all finite")
    diff = float(np.abs(got - want).max())
    if diff > tolerance:
        raise ExportFailed(f"onnxruntime's logits differ from the model's by {diff:.3g}, over {tolerance}")


def export(model, path):
    """Writes the model to `path` as ONNX, checked; returns its size in bytes."""
    data = to_onnx(model)
    verify(model, data)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
