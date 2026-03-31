from typing import Optional
import torch
import torch.nn as nn

ENGINE_MODULES = {
    "cpu": "methods.turboquant.turboquant_cpu",
    "pytorch": "methods.turboquant.turboquant_cpu",
    "triton": "methods.turboquant.turboquant_cpu",
}

_engine_cache = {}


def _get_engine_module(engine: str):
    if engine not in _engine_cache:
        import importlib

        _engine_cache[engine] = importlib.import_module(ENGINE_MODULES[engine])
    return _engine_cache[engine]


def TurboQuantMSE(
    d: int, bits: int, seed: int = 42, device: str = "cpu", engine: Optional[str] = None
):
    if engine is None:
        engine = "cpu"
    mod = _get_engine_module(engine)
    return mod.TurboQuantMSE(d=d, bits=bits, seed=seed, device=device)


def TurboQuantProd(
    d: int,
    bits: int,
    qjl_dim: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "cpu"
    mod = _get_engine_module(engine)
    return mod.TurboQuantProd(d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device)


def TurboQuantKVCache(
    d_key: int,
    d_value: int,
    bits: int = 3,
    seed: int = 42,
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "cpu"
    mod = _get_engine_module(engine)
    return mod.TurboQuantKVCache(
        d_key=d_key, d_value=d_value, bits=bits, seed=seed, device=device
    )
