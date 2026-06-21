"""src.quality — domain exceptions for the compositing + quality pipeline.

The concrete :class:`VideoQualityChecker` lives at
``src.pipeline.quality`` (per Change 2 of
``docs/REPOSITORY_SLIMMING_PLAN.md`` §4 + the QC-extension follow-up).
This package keeps the domain exceptions shared across the compositing
and quality paths.
"""

from src.quality.exceptions import CompositeError, EncodingError, QualityError

__all__ = ["CompositeError", "EncodingError", "QualityError"]
