"""Top-level package for Kaczmarz Algorithms."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.8.0"

__author__ = "Jacob Moorman"
__email__ = "jacob@moorman.me"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Jacob Moorman"


from ._abc import Base
from ._variants import (  # OrthogonalMaxDistance,
    Cyclic,
    MaxDistance,
    Quantile,
    Random,
    RandomOrthoGraph,
    SampledQuantile,
    SVRandom,
    UniformRandom,
    WindowedQuantile,
)
