from typing import Optional
import torch
import torch.nn as nn

# 5개 엔진 명시적 분리
ENGINE_MODULES = {
    "cpu": "methods.turboquant.turboquant_cpu",  # 순수 numpy/scipy
    "torch_cpu": "methods.turboquant.turboquant_torch",  # PyTorch CPU
    "torch_cuda": "methods.turboquant.turboquant_torch",  # PyTorch CUDA
    "cuda_kernel": "methods.turboquant.turboquant_cuda_kernel",  # CUDA 커널
    "triton": "methods.turboquant.turboquant_triton",  # Triton
}

_engine_cache = {}

_DEVICE_FOR_ENGINE = {
    "cpu": "cpu",
    "torch_cpu": "cpu",
    "torch_cuda": "cuda",
    "cuda_kernel": "cuda",
    "triton": "cuda",
}


def _get_engine_module(engine: str):
    if engine not in ENGINE_MODULES:
        raise ValueError(
            f"Unknown TurboQuant engine: '{engine}'. "
            f"Available engines: {list(ENGINE_MODULES.keys())}"
        )
    if engine not in _engine_cache:
        import importlib

        module_path = ENGINE_MODULES[engine]
        try:
            _engine_cache[engine] = importlib.import_module(module_path)
        except ImportError as e:
            raise NotImplementedError(
                f"Engine '{engine}' module '{module_path}' not implemented yet. "
                f"Error: {e}"
            )
    return _engine_cache[engine]


def TurboQuantMSE(
    d: int,
    bits: int,
    seed: int = 42,
    device: Optional[str] = None,
    engine: str = "torch_cuda",
):
    if engine is None:
        engine = "torch_cuda"
    if device is None:
        device = _DEVICE_FOR_ENGINE.get(engine, "cpu")
    mod = _get_engine_module(engine)
    return mod.TurboQuantMSE(d=d, bits=bits, seed=seed, device=device)


def TurboQuantProd(
    d: int,
    bits: int,
    qjl_dim: Optional[int] = None,
    seed: int = 42,
    device: Optional[str] = None,
    engine: str = "torch_cuda",
):
    if engine is None:
        engine = "torch_cuda"
    if device is None:
        device = _DEVICE_FOR_ENGINE.get(engine, "cpu")
    mod = _get_engine_module(engine)
    return mod.TurboQuantProd(d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device)


def TurboQuantKVCache(
    d_key: int,
    d_value: int,
    bits: int = 3,
    seed: int = 42,
    device: Optional[str] = None,
    engine: str = "torch_cuda",
):
    if engine is None:
        engine = "torch_cuda"
    if device is None:
        device = _DEVICE_FOR_ENGINE.get(engine, "cpu")
    mod = _get_engine_module(engine)
    return mod.TurboQuantKVCache(
        d_key=d_key, d_value=d_value, bits=bits, seed=seed, device=device
    )
