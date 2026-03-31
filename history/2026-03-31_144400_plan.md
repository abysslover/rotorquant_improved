# TurboQuant Factory Refactoring Plan

## Goal
Create a unified factory pattern for vector quantization algorithms for LLM KV cache compression.
- Common interface across all methods (PlanarQuant, IsoQuant, RotorQuant, TurboQuant)
- Backend selection (pytorch, cuda, triton, cpu)
- NotImplementedError for unsupported backend combinations

## Execution Order (reverse of repo history)
1. PlanarQuant → 2. IsoQuant → 3. RotorQuant → 4. TurboQuant

---

## Common Interface Definition

### Base Classes

```python
# interfaces.py

class QuantizerBase(nn.Module):
    """Base class for all MSE quantizers."""
    
    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.seed = seed
        self.device = device
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors.
        Returns: (quantized_tensor, indices_dict)
        """
        raise NotImplementedError
    
    def dequantize(self, indices_dict: dict) -> torch.Tensor:
        """Reconstruct vectors from quantized indices."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Full quantize-dequantize cycle."""
        v_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class ProdQuantizerBase(nn.Module):
    """Base class for inner product quantizers (Stage 1 + Stage 2)."""
    
    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.qjl_dim = qjl_dim or d
        self.seed = seed
        self.device = device
    
    def quantize(self, x: torch.Tensor) -> dict:
        """
        Full quantization with QJL.
        Returns: {'mse_indices': ..., 'qjl_signs': ..., 'residual_norm': ...}
        """
        raise NotImplementedError
    
    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Reconstruct from MSE indices."""
        raise NotImplementedError
    
    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Unbiased inner product estimate."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)


class KVCacheBase:
    """Base class for KV cache wrappers."""
    
    def __init__(self, d_key: int, d_value: int, bits: int = 3,
                 seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.seed = seed
        self.device = device
    
    def append(self, keys: torch.Tensor, values: torch.Tensor):
        """Append keys and values to cache."""
        raise NotImplementedError
    
    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute attention scores."""
        raise NotImplementedError
    
    def get_values(self) -> torch.Tensor:
        """Reconstruct cached values."""
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
```

### Backend Enum

```python
# backends.py

from enum import Enum

class Backend(Enum):
    PYTHON = "python"      # Pure Python/PyTorch (default)
    CUDA = "cuda"          # CUDA kernels (if available)
    TRITON = "triton"      # Triton kernels (if available)
    CPU = "cpu"            # Force CPU execution
    METAL = "metal"        # Apple Metal (future)
```

---

## Factory Pattern

```python
# turboquant_factory.py

class TurboQuantFactory:
    """Factory for creating quantization instances."""
    
    # Registry: method_name -> {backend -> class}
    _registry = {}
    
    @classmethod
    def register(cls, method: str, backend: str, quantizer_cls, prod_cls=None, kvcache_cls=None):
        """Register a method + backend combination."""
        if method not in cls._registry:
            cls._registry[method] = {}
        cls._registry[method][backend] = {
            'quantizer': quantizer_cls,
            'prod': prod_cls,
            'kvcache': kvcache_cls,
        }
    
    @classmethod
    def create_quantizer(cls, method: str, backend: str = "python",
                         d: int = 128, bits: int = 3,
                         seed: int = 42, device: str = "cpu", **kwargs):
        """Create MSE quantizer instance."""
        if method not in cls._registry:
            raise ValueError(f"Unknown method: {method}. Available: {list(cls._registry.keys())}")
        
        if backend not in cls._registry[method]:
            raise NotImplementedError(
                f"Backend '{backend}' not implemented for method '{method}'. "
                f"Available backends: {list(cls._registry[method].keys())}"
            )
        
        config = cls._registry[method][backend]
        QuantizerClass = config.get('quantizer')
        if QuantizerClass is None:
            raise NotImplementedError(
                f"Quantizer not implemented for {method}+{backend}"
            )
        
        return QuantizerClass(d=d, bits=bits, seed=seed, device=device, **kwargs)
    
    @classmethod
    def create_prod(cls, method: str, backend: str = "python",
                    d: int = 128, bits: int = 3, qjl_dim: Optional[int] = None,
                    seed: int = 42, device: str = "cpu", **kwargs):
        """Create Prod quantizer instance (Stage 1 + Stage 2)."""
        if method not in cls._registry:
            raise ValueError(f"Unknown method: {method}")
        
        if backend not in cls._registry[method]:
            raise NotImplementedError(f"Backend '{backend}' not implemented for {method}")
        
        config = cls._registry[method][backend]
        ProdClass = config.get('prod')
        if ProdClass is None:
            raise NotImplementedError(f"Prod quantizer not implemented for {method}+{backend}")
        
        return ProdClass(d=d, bits=bits, qjl_dim=qjl_dim, seed=seed, device=device, **kwargs)
    
    @classmethod
    def create_kvcache(cls, method: str, backend: str = "python",
                       d_key: int = 128, d_value: int = 128, bits: int = 3,
                       seed: int = 42, device: str = "cpu", **kwargs):
        """Create KV cache wrapper."""
        if method not in cls._registry:
            raise ValueError(f"Unknown method: {method}")
        
        if backend not in cls._registry[method]:
            raise NotImplementedError(f"Backend '{backend}' not implemented for {method}")
        
        config = cls._registry[method][backend]
        KVCacheClass = config.get('kvcache')
        if KVCacheClass is None:
            raise NotImplementedError(f"KVCache not implemented for {method}+{backend}")
        
        return KVCacheClass(d_key=d_key, d_value=d_value, bits=bits, seed=seed, device=device, **kwargs)
    
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
```

---

## Implementation Steps

### Step 1: Create interfaces.py with base classes
### Step 2: Create backends.py with Backend enum
### Step 3: Create turboquant_factory.py with TurboQuantFactory
### Step 4: Refactor PlanarQuant
### Step 5: Refactor IsoQuant
### Step 6: Refactor RotorQuant
### Step 7: Refactor TurboQuant
### Step 8: Update benchmarks to use factory
### Step 9: Run benchmarks to verify

---

## Current Status

- [x] Define common interface in plan.md
- [x] Create turboquant_factory.py with factory pattern
- [x] Refactor PlanarQuant to use factory
- [x] Refactor IsoQuant to use factory
- [x] Refactor RotorQuant to use factory
- [x] Refactor TurboQuant to use factory
- [x] Update benchmarks to work with factory
- [x] Run benchmarks to verify changes
- [ ] Update benchmarks to work with factory
- [ ] Run benchmarks to verify changes
