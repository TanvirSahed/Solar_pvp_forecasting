"""Importing this package registers every model. Add new model modules to this list."""
from . import baselines, fusion, graph, hdcf, probabilistic, rfkan, ssm, transformers  # noqa: F401

try:  # optional heavy deps (chronos / timesfm / tirex)
    from . import foundation  # noqa: F401
except ImportError as _e:  # pragma: no cover
    import warnings

    warnings.warn(f"foundation-model wrappers unavailable: {_e}")
