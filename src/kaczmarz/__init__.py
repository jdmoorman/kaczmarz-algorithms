"""Top-level package for Kaczmarz Algorithms."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.4.0"

__author__ = "Jacob Moorman"
__email__ = "jacob@moorman.me"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Jacob Moorman"


from ._abc import Base
from ._variants import Cyclic, MaxDistance

__all__ = ["Base", "Cyclic", "MaxDistance"]
