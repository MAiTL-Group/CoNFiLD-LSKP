from typing import Mapping, Optional

import torch


def configure_torch_runtime(config: Optional[Mapping]):
    """Apply explicitly configured PyTorch runtime flags and report their state."""

    config = config or {}
    if not isinstance(config, Mapping):
        raise TypeError("runtime must be a mapping")

    cudnn_benchmark_explicit = "cudnn_benchmark" in config
    if cudnn_benchmark_explicit:
        torch.backends.cudnn.benchmark = bool(config["cudnn_benchmark"])

    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_benchmark_explicit": cudnn_benchmark_explicit,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }

