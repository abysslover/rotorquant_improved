"""
TurboQuant Factory: Unified factory for creating quantization instances.
"""

from typing import Optional, Dict, Type
import torch
import torch.nn as nn

from .interfaces import QuantizerBase, ProdQuantizerBase, KVCacheBase, Backend


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
    """Register all available method + backend combinations."""
    from .planarquant.planarquant import PlanarQuantMSE, PlanarQuantProd
    from .isoquant.isoquant import IsoQuantMSE, IsoQuantProd
    from .rotorquant.rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
    from .turboquant.turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache

    # PlanarQuant - Python backend only for now
    TurboQuantFactory.register(
        "planarquant",
        "python",
        quantizer_cls=PlanarQuantMSE,
        prod_cls=PlanarQuantProd,
    )

    # IsoQuant - Python backend only for now
    TurboQuantFactory.register(
        "isoquant",
        "python",
        quantizer_cls=IsoQuantMSE,
        prod_cls=IsoQuantProd,
    )

    # RotorQuant - Python backend only for now
    TurboQuantFactory.register(
        "rotorquant",
        "python",
        quantizer_cls=RotorQuantMSE,
        prod_cls=RotorQuantProd,
        kvcache_cls=RotorQuantKVCache,
    )

    # TurboQuant - Python backend only for now
    TurboQuantFactory.register(
        "turboquant",
        "python",
        quantizer_cls=TurboQuantMSE,
        prod_cls=TurboQuantProd,
        kvcache_cls=TurboQuantKVCache,
    )


class TurboQuantProdFactory:
    """
    Factory for creating Prod quantizer instances with method + engine parameters.

    Usage:
        # Create Prod instance with method and engine
        prod = TurboQuantProdFactory.create(
            method="isoquant",
            engine="pytorch",
            d=128, bits=3, seed=42, device="cuda"
        )

    Mapping:
        - method: planarquant, isoquant, rotorquant, turboquant
        - engine: cpu, pytorch, triton
        - cpu -> backend "python" with device="cpu"
        - pytorch -> backend "python" with device="cuda"
        - triton -> backend "triton" (future)
    """

    _ENGINE_TO_BACKEND = {
        "cpu": "python",
        "pytorch": "python",
        "triton": "triton",
    }

    _ENGINE_TO_DEVICE = {
        "cpu": "cpu",
        "pytorch": "cuda",
        "triton": "cuda",
    }

    @classmethod
    def create(
        cls,
        method: str,
        engine: str = "pytorch",
        d: int = 128,
        bits: int = 3,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ) -> ProdQuantizerBase:
        """
        Create Prod quantizer instance.

        Args:
            method: Quantization method ("planarquant", "isoquant", "rotorquant", "turboquant")
            engine: Computation engine ("cpu", "pytorch", "triton")
            d: Vector dimension
            bits: Total bits per component
            qjl_dim: QJL projection dimension
            seed: Random seed
            device: Override device (if provided, ignores engine mapping)
            **kwargs: Additional method-specific arguments

        Returns:
            ProdQuantizerBase instance
        """
        backend = cls._ENGINE_TO_BACKEND.get(engine, "python")

        if device is None:
            device = cls._ENGINE_TO_DEVICE.get(engine, "cpu")

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
        engine: str = "pytorch",
        d: int = 128,
        bits: int = 3,
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs,
    ) -> QuantizerBase:
        """
        Create MSE quantizer instance (Stage 1 only).

        Args:
            method: Quantization method
            engine: Computation engine
            d: Vector dimension
            bits: Bits per component
            seed: Random seed
            device: Override device
            **kwargs: Additional arguments

        Returns:
            QuantizerBase instance
        """
        backend = cls._ENGINE_TO_BACKEND.get(engine, "python")

        if device is None:
            device = cls._ENGINE_TO_DEVICE.get(engine, "cpu")

        return TurboQuantFactory.create_quantizer(
            method=method,
            backend=backend,
            d=d,
            bits=bits,
            seed=seed,
            device=device,
            **kwargs,
        )


# Auto-register on import
register_all_methods()
