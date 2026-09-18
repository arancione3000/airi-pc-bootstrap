"""Airi-PC neuroevolution subsystem.

The package is intentionally import-light: importing :mod:`evolution` does not
import torch, so the core Airi-PC runtime stays small until evolution is used.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
