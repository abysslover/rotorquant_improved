from typing import Optional, Tuple
import torch
import torch.nn as nn

ENGINE_MODULES = {
    "cpu": "methods.planarquant.planarquant_cpu",
    "pytorch": "methods.planarquant.planarquant_pytorch",
    "triton": "methods.planarquant.planarquant_triton",
}

_engine_cache = {}


def _get_engine_module(engine: str):
    if engine not in _engine_cache:
        import importlib

        _engine_cache[engine] = importlib.import_module(ENGINE_MODULES[engine])
    return _engine_cache[engine]


def PlanarQuantMSE(
    d: int, bits: int, seed: int = 42, device: str = "cpu", engine: Optional[str] = None
):
    if engine is None:
        engine = "pytorch" if device.startswith("cuda") else "cpu"
    mod = _get_engine_module(engine)
    return mod.PlanarQuantMSE(d=d, bits=bits, seed=seed, device=device)


def PlanarQuantProd(
    d: int,
    bits: int,
    qjl_dim: Optional[int] = None,
    seed: int = 42,
    device: str = "cpu",
    engine: Optional[str] = None,
):
    if engine is None:
        engine = "pytorch" if device.startswith("cuda") else "cpu"
    mod = _get_engine_module(engine)
    return mod.PlanarQuantProd(
        d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device
    )
