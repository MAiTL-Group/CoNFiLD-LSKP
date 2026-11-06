# CoNFiLD-LSKP modification notice (2026):
# Added single-process and optional distributed execution support.
# Derived from CoNFiLD; see third_party_licenses/README.md for attribution.
"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3

_MPI = None
_SINGLE_PROCESS = False


def _get_mpi():
    """Import MPI only for recipes that explicitly use MPI launching."""
    global _MPI
    if _MPI is None:
        from mpi4py import MPI

        _MPI = MPI
    return _MPI


def setup_dist(use_mpi=True):
    """
    Setup a distributed process group.

    Single-GPU recipes use a one-rank torch process group and do not import
    MPI. Multi-process recipes use MPI for distributed initialization.
    """
    global _SINGLE_PROCESS
    if dist.is_initialized():
        return
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if not use_mpi:
        _SINGLE_PROCESS = True
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{_find_free_port()}",
            rank=0,
            world_size=1,
        )
        return

    MPI = _get_mpi()
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[MPI.COMM_WORLD.Get_rank()]

    comm = MPI.COMM_WORLD

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if _SINGLE_PROCESS:
        with bf.BlobFile(path, "rb") as f:
            return th.load(io.BytesIO(f.read()), **kwargs)

    MPI = _get_mpi()
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
