"""Neural-network building blocks with hand-written forward/backward passes.

File layout mirrors ``torch/nn/modules/`` (linear, activation, conv, pooling,
batchnorm, dropout, loss, init) so the names are already familiar; ``blocks``
holds the fused convenience layers.
"""
from .activation import ReLU
from .batchnorm import BatchNorm, SpatialBatchNorm
from .blocks import (
    Conv_BatchNorm_ReLU,
    Conv_BatchNorm_ReLU_Pool,
    Conv_ReLU,
    Conv_ReLU_Pool,
    Linear_BatchNorm_ReLU,
    Linear_ReLU,
)
from .conv import Conv, FastConv
from .dropout import Dropout
from .init import kaiming_initializer
from .linear import Linear
from .loss import SoftmaxCrossEntropy
from .pooling import FastMaxPool, MaxPool

__all__ = [
    # primitives
    "Linear",
    "ReLU",
    "Conv",
    "MaxPool",
    "BatchNorm",
    "SpatialBatchNorm",
    "Dropout",
    # fast reference implementations
    "FastConv",
    "FastMaxPool",
    # fused layers
    "Linear_ReLU",
    "Conv_ReLU",
    "Conv_ReLU_Pool",
    "Linear_BatchNorm_ReLU",
    "Conv_BatchNorm_ReLU",
    "Conv_BatchNorm_ReLU_Pool",
    # loss and init
    "SoftmaxCrossEntropy",
    "kaiming_initializer",
]
