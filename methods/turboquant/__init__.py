from .turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
from .lloyd_max import LloydMaxCodebook, solve_lloyd_max
from .compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from .cuda_backend import is_cuda_available, QJLSketch, QJLKeyQuantizer
from ..isoquant import IsoQuantMSE, IsoQuantProd
from ..planarquant import PlanarQuantMSE, PlanarQuantProd
from ..rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
from ..rotorquant.clifford import (
    geometric_product,
    make_random_rotor,
    rotor_sandwich,
)

QuantMSE = IsoQuantMSE
QuantProd = IsoQuantProd

try:
    from .triton_planarquant import (
        triton_planar2_fused,
        triton_planar2_quantize,
        triton_planar2_dequantize,
    )
    from .fused_planar_attention import (
        triton_fused_planar_quantize_attend,
        triton_planar_cached_attention,
        pre_rotate_query_planar,
        PlanarQuantCompressedCache,
    )
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
