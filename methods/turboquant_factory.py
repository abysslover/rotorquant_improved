"""
TurboQuant Factory: Unified factory for creating quantization instances.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Type, Union
import torch
import torch.nn as nn

from .interfaces import QuantizerBase, ProdQuantizerBase, KVCacheBase, Backend


# Global CUDA availability flag
_CUDA_AVAILABLE = None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = torch.cuda.is_available()
    return _CUDA_AVAILABLE


def ensure_cuda_kernels_built() -> bool:
    """
    CUDA kernels가 없으면 자동으로 빌드를 시도합니다.
    """
    if is_cuda_available():
        return True

    # Check if already built
    try:
        from .turboquant import cuda_backend

        if (
            hasattr(cuda_backend, "is_cuda_available")
            and cuda_backend.is_cuda_available()
        ):
            return True
    except ImportError:
        pass

    csrc_dir = Path(__file__).resolve().parent / "turboquant" / "csrc"
    setup_py = (
        Path(__file__).resolve().parent.parent / "turboquant" / "csrc" / "setup.py"
    )

    # Check if setup.py exists
    if not setup_py.exists():
        print(f"[INFO] CUDA csrc not found at {csrc_dir}")
        return False

    # CUDA_HOME 자동 설정
    if "CUDA_HOME" not in os.environ:
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.0",
            "/opt/cuda",
        ]
        for path in cuda_paths:
            if os.path.exists(os.path.join(path, "bin", "nvcc")):
                os.environ["CUDA_HOME"] = path
                print(f"[INFO] Auto-detected CUDA_HOME={path}")
                break

    print(
        f"[INFO] Building CUDA kernels with CUDA_HOME={os.environ.get('CUDA_HOME', 'NOT SET')}"
    )

    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(csrc_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        print("[SUCCESS] CUDA kernels built successfully")

        # Reload module after build
        import importlib

        try:
            importlib.reload(sys.modules.get("methods.turboquant.cuda_backend", None))
        except:
            pass

        return is_cuda_available()

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] CUDA kernel build failed:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] CUDA kernel build error: {e}")
        return False


class TurboQuantFactory:
    """
    Factory for creating vector quantization instances.

    Usage:
        # Create MSE quantizer
        quantizer = TurboQuantFactory.create_quantizer(
            method="isoquant",
            backend="python",
            d=128, bits=3, seed=42, device="cuda"
        )

        # Create Prod quantizer (Stage 1 + Stage 2)
        prod = TurboQuantFactory.create_prod(
            method="isoquant",
            backend="python",
            d=128, bits=3, seed=42, device="cuda"
        )

        # Create KV cache wrapper
        cache = TurboQuantFactory.create_kvcache(
            method="isoquant",
            backend="python",
            d_key=128, d_value=128, bits=3, device="cuda"
        )
    """

    _registry: Dict[str, Dict[str, dict]] = {}

    @classmethod
    def register(
        cls,
        method: str,
        backend: str,
        quantizer_cls: Optional[Type[nn.Module]] = None,
        prod_cls: Optional[Type[nn.Module]] = None,
        kvcache_cls: Optional[Type[KVCacheBase]] = None,
    ):
        """
        Register a method + backend combination.

        Args:
            method: Method name (e.g., "planarquant", "isoquant", "rotorquant", "turboquant")
            backend: Backend name (e.g., "python", "cuda", "triton")
            quantizer_cls: MSE quantizer class (Stage 1 only)
            prod_cls: Prod quantizer class (Stage 1 + Stage 2)
            kvcache_cls: KV cache wrapper class
        """
        if method not in cls._registry:
            cls._registry[method] = {}
        cls._registry[method][backend] = {
            "quantizer": quantizer_cls,
            "prod": prod_cls,
            "kvcache": kvcache_cls,
        }

    @classmethod
    def create_quantizer(
        cls,
        method: str,
        backend: str = "python",
        d: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ) -> QuantizerBase:
        """
        Create MSE quantizer instance (Stage 1 only).

        Args:
            method: Quantization method ("planarquant", "isoquant", "rotorquant", "turboquant")
            backend: Computation backend ("python", "cuda", "triton", "cpu")
            d: Vector dimension
            bits: Bits per component for Lloyd-Max quantization
            seed: Random seed for reproducibility
            device: Torch device
            **kwargs: Additional method-specific arguments

        Returns:
            QuantizerBase instance

        Raises:
            ValueError: If method is unknown
            NotImplementedError: If backend not supported for method
        """
        if method not in cls._registry:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(cls._registry.keys())}"
            )

        if backend not in cls._registry[method]:
            raise NotImplementedError(
                f"Backend '{backend}' not implemented for method '{method}'. "
                f"Available backends: {list(cls._registry[method].keys())}"
            )

        config = cls._registry[method][backend]
        QuantizerClass = config.get("quantizer")
        if QuantizerClass is None:
            raise NotImplementedError(
                f"Quantizer not implemented for {method}+{backend}"
            )

        return QuantizerClass(d=d, bits=bits, seed=seed, device=device, **kwargs)

    @classmethod
    def create_prod(
        cls,
        method: str,
        backend: str = "python",
        d: int = 128,
        bits: int = 3,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ) -> ProdQuantizerBase:
        """
        Create Prod quantizer instance (Stage 1 + Stage 2 with QJL).

        Args:
            method: Quantization method
            backend: Computation backend
            d: Vector dimension
            bits: Total bits per component (MSE uses bits-1, QJL uses 1)
            qjl_dim: QJL projection dimension (default = d)
            seed: Random seed
            device: Torch device
            **kwargs: Additional method-specific arguments

        Returns:
            ProdQuantizerBase instance

        Raises:
            ValueError: If method is unknown
            NotImplementedError: If backend not supported
        """
        if method not in cls._registry:
            raise ValueError(f"Unknown method: {method}")

        if backend not in cls._registry[method]:
            raise NotImplementedError(
                f"Backend '{backend}' not implemented for {method}"
            )

        config = cls._registry[method][backend]
        ProdClass = config.get("prod")
        if ProdClass is None:
            raise NotImplementedError(
                f"Prod quantizer not implemented for {method}+{backend}"
            )

        return ProdClass(
            d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device, **kwargs
        )

    @classmethod
    def create_kvcache(
        cls,
        method: str,
        backend: str = "python",
        d_key: int = 128,
        d_value: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ) -> KVCacheBase:
        """
        Create KV cache wrapper.

        Args:
            method: Quantization method
            backend: Computation backend
            d_key: Key dimension
            d_value: Value dimension
            bits: Bits per component
            seed: Random seed
            device: Torch device
            **kwargs: Additional method-specific arguments

        Returns:
            KVCacheBase instance

        Raises:
            ValueError: If method is unknown
            NotImplementedError: If backend not supported
        """
        if method not in cls._registry:
            raise ValueError(f"Unknown method: {method}")

        if backend not in cls._registry[method]:
            raise NotImplementedError(
                f"Backend '{backend}' not implemented for {method}"
            )

        config = cls._registry[method][backend]
        KVCacheClass = config.get("kvcache")
        if KVCacheClass is None:
            raise NotImplementedError(f"KVCache not implemented for {method}+{backend}")

        return KVCacheClass(
            d_key=d_key, d_value=d_value, bits=bits, seed=seed, device=device, **kwargs
        )

    @classmethod
    def available_methods(cls) -> list:
        """List all available methods."""
        return list(cls._registry.keys())

    @classmethod
    def available_backends(cls, method: str) -> list:
        """List all available backends for a method."""
        if method not in cls._registry:
            return []
        return list(cls._registry[method].keys())


def register_all_methods():
    from .planarquant.planarquant_cpu import PlanarQuantMSE as PQCPUMSE
    from .planarquant.planarquant_cpu import PlanarQuantProd as PQCPUProd
    from .planarquant.planarquant_pytorch import PlanarQuantMSE as PQPyTorchMSE
    from .planarquant.planarquant_pytorch import PlanarQuantProd as PQPyTorchProd
    from .planarquant.planarquant_triton import PlanarQuantMSE as PQTritonMSE
    from .planarquant.planarquant_triton import PlanarQuantProd as PQTritonProd

    from .isoquant.isoquant_cpu import IsoQuantMSE as IQCPUMSE
    from .isoquant.isoquant_cpu import IsoQuantProd as IQCPUProd
    from .isoquant.isoquant_triton import IsoQuantMSE as IQTritonMSE
    from .isoquant.isoquant_triton import IsoQuantProd as IQTritonProd

    from .rotorquant.rotorquant_cpu import RotorQuantMSE as RQCPUMSE
    from .rotorquant.rotorquant_cpu import RotorQuantProd as RQCPUProd
    from .rotorquant.rotorquant import RotorQuantMSE as RQPyTorchMSE
    from .rotorquant.rotorquant import RotorQuantProd as RQPyTorchProd
    from .rotorquant.rotorquant import RotorQuantKVCache
    from .rotorquant.rotorquant_triton import RotorQuantMSE as RQTritonMSE
    from .rotorquant.rotorquant_triton import RotorQuantProd as RQTritonProd

    from .turboquant.turboquant_cpu import TurboQuantMSE as TQCPUMSE
    from .turboquant.turboquant_cpu import TurboQuantProd as TQCPUProd
    from .turboquant.turboquant import TurboQuantMSE as TQPyTorchMSE
    from .turboquant.turboquant import TurboQuantProd as TQPyTorchProd
    from .turboquant.turboquant import TurboQuantKVCache

    TurboQuantFactory.register(
        "planarquant", "python", quantizer_cls=PQCPUMSE, prod_cls=PQCPUProd
    )
    TurboQuantFactory.register(
        "planarquant", "cuda", quantizer_cls=PQPyTorchMSE, prod_cls=PQPyTorchProd
    )
    TurboQuantFactory.register(
        "planarquant", "triton", quantizer_cls=PQTritonMSE, prod_cls=PQTritonProd
    )

    TurboQuantFactory.register(
        "isoquant", "python", quantizer_cls=IQCPUMSE, prod_cls=IQCPUProd
    )
    TurboQuantFactory.register(
        "isoquant", "triton", quantizer_cls=IQTritonMSE, prod_cls=IQTritonProd
    )

    TurboQuantFactory.register(
        "rotorquant", "python", quantizer_cls=RQCPUMSE, prod_cls=RQCPUProd
    )
    TurboQuantFactory.register(
        "rotorquant",
        "cuda",
        quantizer_cls=RQPyTorchMSE,
        prod_cls=RQPyTorchProd,
        kvcache_cls=RotorQuantKVCache,
    )
    TurboQuantFactory.register(
        "rotorquant", "triton", quantizer_cls=RQTritonMSE, prod_cls=RQTritonProd
    )

    TurboQuantFactory.register(
        "turboquant", "python", quantizer_cls=TQCPUMSE, prod_cls=TQCPUProd
    )
    TurboQuantFactory.register(
        "turboquant",
        "cuda",
        quantizer_cls=TQPyTorchMSE,
        prod_cls=TQPyTorchProd,
        kvcache_cls=TurboQuantKVCache,
    )


class TurboQuantProdFactory:
    """
    Factory for creating Prod quantizer instances with method + engine parameters.

    Usage:
        # Create Prod instance with method and engine
        prod = TurboQuantProdFactory.create(
            method="isoquant",
            engine="torch_cuda",
            d=128, bits=3, seed=42, device="cuda"
        )

    Mapping:
        - method: planarquant, isoquant, rotorquant, turboquant
        - engine: cpu, torch_cpu, cuda_kernel, torch_cuda, triton
        - cpu -> backend "python" with device="cpu" (pure Python)
        - torch_cpu -> backend "python" with device="cpu" (PyTorch, CPU)
        - cuda_kernel -> backend "cuda" with device="cuda" (compiled CUDA)
        - torch_cuda -> backend "python" with device="cuda" (PyTorch, CUDA)
        - triton -> backend "triton" with device="cuda"
    """

    _ENGINE_TO_BACKEND = {
        "cpu": "python",
        "torch_cpu": "python",
        "cuda_kernel": "cuda",
        "torch_cuda": "python",
        "triton": "triton",
    }

    _ENGINE_TO_DEVICE = {
        "cpu": "cpu",
        "torch_cpu": "cpu",
        "cuda_kernel": "cuda",
        "torch_cuda": "cuda",
        "triton": "cuda",
    }

    _ENGINE_ALIASES = {
        "pytorch": "torch_cuda",
    }

    @classmethod
    def _resolve_engine(cls, engine: str) -> str:
        return cls._ENGINE_ALIASES.get(engine, engine)

    @classmethod
    def create(
        cls,
        method: str,
        engine: str = "torch_cuda",
        d: int = 128,
        bits: int = 3,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ) -> ProdQuantizerBase:
        engine = cls._resolve_engine(engine)
        backend = cls._ENGINE_TO_BACKEND.get(engine)

        if backend is None:
            raise NotImplementedError(
                f"Engine '{engine}' not supported. "
                f"Available engines: {list(cls._ENGINE_TO_BACKEND.keys())}"
            )

        if device is None:
            device = cls._ENGINE_TO_DEVICE.get(engine, "cpu")

        if backend == "cuda":
            if not ensure_cuda_kernels_built():
                raise NotImplementedError(
                    f"CUDA backend not available for {method}+{engine}. "
                    "CUDA kernels failed to build."
                )

        return TurboQuantFactory.create_prod(
            method=method,
            backend=backend,
            d=d,
            bits=bits,
            qjl_dim=qjl_dim,
            seed=seed,
            device=device,
            **kwargs,
        )

    @classmethod
    def create_quantizer(
        cls,
        method: str,
        engine: str = "torch_cuda",
        d: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ) -> QuantizerBase:
        engine = cls._resolve_engine(engine)
        backend = cls._ENGINE_TO_BACKEND.get(engine)

        if backend is None:
            raise NotImplementedError(
                f"Engine '{engine}' not supported. "
                f"Available engines: {list(cls._ENGINE_TO_BACKEND.keys())}"
            )

        if device is None:
            device = cls._ENGINE_TO_DEVICE.get(engine, "cpu")

        if backend == "cuda":
            if not ensure_cuda_kernels_built():
                raise NotImplementedError(
                    f"CUDA backend not available for {method}+{engine}. "
                    "CUDA kernels failed to build."
                )

        return TurboQuantFactory.create_quantizer(
            method=method,
            backend=backend,
            d=d,
            bits=bits,
            seed=seed,
            device=device,
            **kwargs,
        )

    @classmethod
    def available_engines(cls) -> list:
        return list(cls._ENGINE_TO_BACKEND.keys())


# Auto-register on import
register_all_methods()
