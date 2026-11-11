"""
The speedrun's model contract (SPEEDRUN.md), checked on the bytes of an exported model.

Locked: the harness (eval/check.py, eval/score.py) loads this file by its path in the locked tree,
and the training script's export (nanochat/export.py) imports it, so that a model the gate would
refuse fails the round itself, with a message agents read. It imports onnx alone, never nanochat.

The contract:
- one int64 input [1, T_EVAL] (token ids) and one float output [1, T_EVAL, V], V >= VOCAB (logits;
  the columns past VOCAB are ignored);
- only ops of ONNX's default domain, and none that draws random numbers;
- no external data, and no local functions;
- at most MAX_MODEL_BYTES in all.
"""

import onnx
from onnx import AttributeProto, TensorProto

# The tokens a model reads at once, and the tokenizer's vocabulary (bench/tokenizer).
T_EVAL = 512
VOCAB = 8192

# An absolute cap, under protobuf's 2 GiB. The gate also caps a model at SIZE_CAP times the root's
# bytes, and its evaluation at TIME_CAP times the root's time, so that no training moves into the
# untimed evaluation.
MAX_MODEL_BYTES = 512 * 1024 * 1024
SIZE_CAP = 8
TIME_CAP = 8

DEFAULT_DOMAINS = frozenset({"", "ai.onnx"})

# Ops whose output is random: they would make the evaluation, and the causality probe, depend on
# the draw. Dropout is random when its training_mode input is set.
RANDOM_OPS = frozenset({
    "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
    "Multinomial", "Bernoulli", "Dropout",
})


def _external(tensor):
    return tensor.data_location == TensorProto.EXTERNAL or len(tensor.external_data) > 0


def _graph_problems(graph, where):
    problems = []
    for tensor in graph.initializer:
        if _external(tensor):
            problems.append(f"{where}: initializer {tensor.name!r} has external data")
    for sparse in graph.sparse_initializer:
        for tensor in (sparse.values, sparse.indices):
            if _external(tensor):
                problems.append(f"{where}: sparse initializer {tensor.name!r} has external data")
    for node in graph.node:
        label = f"{where}: node {node.name or node.op_type!r} ({node.domain or 'ai.onnx'}::{node.op_type})"
        if node.domain not in DEFAULT_DOMAINS:
            problems.append(f"{label}: domain {node.domain!r} is not ONNX's default domain")
        if node.op_type in RANDOM_OPS:
            problems.append(f"{label}: {node.op_type} is random")
        for attr in node.attribute:
            tensors = []
            if attr.type == AttributeProto.TENSOR:
                tensors = [attr.t]
            elif attr.type == AttributeProto.TENSORS:
                tensors = list(attr.tensors)
            elif attr.type == AttributeProto.SPARSE_TENSOR:
                tensors = [attr.sparse_tensor.values, attr.sparse_tensor.indices]
            elif attr.type == AttributeProto.SPARSE_TENSORS:
                tensors = [t for s in attr.sparse_tensors for t in (s.values, s.indices)]
            for tensor in tensors:
                if _external(tensor):
                    problems.append(f"{label}: attribute {attr.name!r} has external data")
            if attr.type == AttributeProto.GRAPH:
                problems += _graph_problems(attr.g, f"{label}/{attr.name}")
            elif attr.type == AttributeProto.GRAPHS:
                for i, g in enumerate(attr.graphs):
                    problems += _graph_problems(g, f"{label}/{attr.name}[{i}]")
    return problems


def _dims(value_info):
    """The static dims of a tensor's type, or None where a dim is symbolic or the type isn't a tensor."""
    if not value_info.type.HasField("tensor_type"):
        return None
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if not d.HasField("dim_value"):
            return None
        dims.append(d.dim_value)
    return dims


def problems(model_bytes, t_eval=T_EVAL, vocab=VOCAB):
    """What breaks the contract in the model `model_bytes`, each said in a line: none if it keeps it."""
    if len(model_bytes) > MAX_MODEL_BYTES:
        return [f"the model is {len(model_bytes)} bytes, over {MAX_MODEL_BYTES}"]
    try:
        model = onnx.load_model_from_string(model_bytes)
    except Exception as e:  # a protobuf that doesn't parse
        return [f"the model is no ONNX model: {type(e).__name__}: {e}"]
    found = []
    for opset in model.opset_import:
        if opset.domain not in DEFAULT_DOMAINS:
            found.append(f"opset domain {opset.domain!r} is not ONNX's default domain")
    if len(model.functions) > 0:
        found.append(f"the model has {len(model.functions)} local functions; export them inlined")
    found += _graph_problems(model.graph, "graph")
    initializers = {t.name for t in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in initializers]
    if len(inputs) != 1:
        found.append(f"the model has {len(inputs)} inputs, not 1")
    else:
        elem = inputs[0].type.tensor_type.elem_type
        dims = _dims(inputs[0])
        if elem != TensorProto.INT64 or dims != [1, t_eval]:
            found.append(f"the input is {TensorProto.DataType.Name(elem)} {dims}, not INT64 [1, {t_eval}]")
    outputs = list(model.graph.output)
    if len(outputs) != 1:
        found.append(f"the model has {len(outputs)} outputs, not 1")
    else:
        elem = outputs[0].type.tensor_type.elem_type
        dims = _dims(outputs[0])
        if elem != TensorProto.FLOAT or dims is None or len(dims) != 3 or dims[:2] != [1, t_eval] or dims[2] < vocab:
            found.append(f"the output is {TensorProto.DataType.Name(elem)} {dims}, not FLOAT [1, {t_eval}, V] with V >= {vocab}")
    return found
