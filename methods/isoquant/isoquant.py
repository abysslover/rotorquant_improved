from typing import Optional
import torch
import torch.nn as nn

ENGINE_MODULES = {
    "cpu": "methods.isoquant.isoquant_cpu",
    "pytorch": "methods.isoquant.isoquant_cpu",
    "triton": "methods.isoquant.isoquant_triton",
}

_engine_cache = {}


def _get_engine_module(engine: str):
    if engine not in _engine_cache:
        import importlib

        _engine_cache[engine] = importlib.import_module(ENGINE_MODULES[engine])
    return _engine_cache[engine]


def IsoQuantMSE(
    d: int,
    bits: int,
    seed: int = 42,
    mode: str = "full",
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "triton" if device.startswith("cuda") else "cpu"
    mod = _get_engine_module(engine)
    return mod.IsoQuantMSE(d=d, bits=bits, seed=seed, mode=mode, device=device)


def IsoQuantProd(
    d: int,
    bits: int,
    mode: str = "full",
    qjl_dim: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "triton" if device.startswith("cuda") else "cpu"
    mod = _get_engine_module(engine)
    return mod.IsoQuantProd(
        d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, mode=mode, device=device
    )
