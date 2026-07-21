"""Dataset loaders — one module per dataset (cifar, mnist, ...).

Each module exposes a raw loader and a preprocessing helper; downloads land in
``data/raw/``.
"""
from .cifar import CLASSES, cifar10, preprocess_cifar10

__all__ = ["CLASSES", "cifar10", "preprocess_cifar10"]
