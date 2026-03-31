import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

ENGINE_MODULES = {
    "cpu": "methods.rotorquant.rotorquant_cpu",
    "pytorch": "methods.rotorquant.rotorquant_cpu",
    "triton": "methods.rotorquant.rotorquant_cpu",
}

_engine_cache = {}


def _get_engine_module(engine: str):
    if engine not in _engine_cache:
        import importlib

        _engine_cache[engine] = importlib.import_module(ENGINE_MODULES[engine])
    return _engine_cache[engine]


def RotorQuantMSE(
    d: int,
    bits: int,
    seed: int = 42,
    grade_bits: Optional[Dict[str, int]] = None,
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "cpu"
    mod = _get_engine_module(engine)
    return mod.RotorQuantMSE(
        d=d, bits=bits, seed=seed, grade_bits=grade_bits, device=device
    )


def RotorQuantProd(
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
    return mod.RotorQuantProd(d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device)


def RotorQuantKVCache(
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
    return mod.RotorQuantKVCache(
        d_key=d_key, d_value=d_value, bits=bits, seed=seed, device=device
    )
