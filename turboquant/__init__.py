from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer
from .isoquant import IsoQuantMSE, IsoQuantProd
from .rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
from .clifford import geometric_product, make_random_rotor, rotor_sandwich

# IsoQuant is the recommended default (5.8x faster, same quality)
QuantMSE = IsoQuantMSE
QuantProd = IsoQuantProd

# Triton kernels (optional, requires triton >= 3.0)
try:
    from .triton_isoquant import (
        triton_iso_full_fused,
        triton_iso_fast_fused,
    )
    from .triton_kernels import (
        triton_rotor_sandwich,
        triton_rotor_full_fused,
        triton_rotor_inverse_sandwich,
        triton_fused_attention,
        pack_rotors_for_triton,
    )
    _triton_available = True
except ImportError:
    _triton_available = False
