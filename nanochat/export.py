"""
Exports the trained model as ONNX, the speedrun's only output (SPEEDRUN.md), and checks it as the
gate will: the contract in nanochat/onnx_rules.py (locked), then onnxruntime against the model's own
logits. A model that fails raises ExportFailed, so the round crashes with the reason in its log, which
agents read; the gate's own reasons never reach them.

A snapshot (plan 16e of pi-autoresearch-serverless) is an export's graph with the model's current
weights written into it, by name: no new export, so it takes a fraction of a second. The harness
scores a snapshot only if its graph is the final model's, but for its weights' values and its
metadata; the export runs no graph optimizer, which would fold weights into constants of other names.

Each snapshot, and the final model, records how many tokens it trained on, as the metadata entry
`tokens` (plan 16f): the harness's learning curves have one by time and one by tokens trained.

Editable: change how the model exports, as long as what it writes keeps the contract.
"""

import logging
import os

import numpy as np

# onnxruntime sends telemetry to Microsoft from a thread of its own on Linux: a network socket, which
# the sandbox denies, failing the round. Off before it loads (the harness's command sets it too).
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from nanochat.onnx_rules import T_EVAL, VOCAB, problems  # noqa: E402

for _name in ("onnxscript", "onnx_ir", "torch.onnx"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# The metadata entry the harness reads a model's tokens trained from (SPEEDRUN.md).
TOKENS = "tokens"


class ExportFailed(Exception):
    pass


def record_tokens(proto, tokens):
    """Writes `tokens`, the tokens the model trained on, into the model's metadata, as the harness
    reads them; nothing when it is None."""
    if tokens is None:
        return
    for prop in proto.metadata_props:
        if prop.key == TOKENS:
            prop.value = str(tokens)
            return
    proto.metadata_props.add(key=TOKENS, value=str(tokens))


def to_onnx(model, tokens=None):
    """The model's bytes as ONNX: one int64 input [1, T_EVAL], one float output [1, T_EVAL, VOCAB],
    and the tokens it trained on in its metadata, when given. Export the plain module, never a
    torch.compile'd one. With no graph optimizer (onnxruntime folds constants itself as it loads a
    model), every weight is an initializer named after its parameter, which a snapshot fills in."""
    was_training = model.training
    model.eval()
    idx = torch.zeros((1, T_EVAL), dtype=torch.long)
    try:
        with torch.no_grad():
            program = torch.onnx.export(
                model, (idx,), dynamo=True, opset_version=18,
                input_names=["idx"], output_names=["logits"], verbose=False, optimize=False,
            )
        proto = program.model_proto
        record_tokens(proto, tokens)
        return proto.SerializeToString()
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


def template(data):
    """An export, parsed, for snapshots to fill in: the model's at any weights, as long as its
    parameters' names and shapes are the final model's."""
    return onnx.load_model_from_string(data)


def snapshot(template, model, tokens=None):
    """The template's graph with the model's current weights, written into the initializers named after
    its parameters and buffers, and the tokens it trained on: the export of the same weights and
    tokens, byte for byte."""
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    proto = onnx.ModelProto()
    proto.CopyFrom(template)
    for init in proto.graph.initializer:
        tensor = current.get(init.name)
        if tensor is not None:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            array = tensor.detach().to("cpu").float().numpy() if tensor.is_floating_point() else tensor.detach().to("cpu").numpy()
            init.CopyFrom(numpy_helper.from_array(array.astype(dtype, copy=False), init.name))
    record_tokens(proto, tokens)
    return proto.SerializeToString()


def export(model, path, tokens=None):
    """Writes the model to `path` as ONNX, checked, with the tokens it trained on; returns its size in
    bytes."""
    data = to_onnx(model, tokens)
    verify(model, data)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
