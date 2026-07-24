import torch

from .activation import ReLU
from .batchnorm import BatchNorm, SpatialBatchNorm
from .conv import FastConv
from .linear import Linear
from .pooling import FastMaxPool


class Linear_ReLU:
    """Linear followed by ReLU"""
    
    @staticmethod
    def forward(
        x   :   torch.Tensor,
        w   :   torch.Tensor,
        b   :   torch.Tensor
    )-> tuple[torch.Tensor, tuple]:
        fc_out, fc_cache = Linear.forward(x, w, b)
        relu_out, relu_cache  = ReLU.forward(fc_out)
        cache = (fc_cache, relu_cache)
        return relu_out, cache

    @staticmethod
    def backward(
        dout    : torch.Tensor,
        cache   : tuple
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fc_cache, relu_cache = cache
        fc_dout = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(fc_dout, fc_cache)
        return dx, dw, db


class Conv_ReLU:
    """Convolution followed by a ReLU."""

    @staticmethod
    def forward(x, w, b, conv_param):
        conv_out, conv_cache = FastConv.forward(x, w, b, conv_param)
        relu_out, relu_cache = ReLU.forward(conv_out)
        cache = (conv_cache, relu_cache)
        return relu_out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache = cache
        dconv_out = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(dconv_out, conv_cache)
        return dx, dw, db
class Conv_ReLU_Pool:
    """Convolution, ReLU, then max pooling."""

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        conv_out, conv_cache = FastConv.forward(x, w, b, conv_param)
        relu_out, relu_cache = ReLU.forward(conv_out)
        maxpool_out, maxpool_cache = FastMaxPool.forward(relu_out, pool_param)
        cache = (conv_cache, relu_cache, maxpool_cache)
        return maxpool_out, cache
    
    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache, maxpool_cache = cache
        drelu_out = FastMaxPool.backward(dout, maxpool_cache)
        dconv_out = ReLU.backward(drelu_out, relu_cache)
        dx, dw, db = FastConv.backward(dconv_out, conv_cache)
        
        return dx, dw, db


class Linear_BatchNorm_ReLU:
    """Affine transform, batch normalization, then a ReLU."""

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        linear_out, linear_cache = Linear.forward(x, w, b)
        batchnorm_out, batchnorm_cache = BatchNorm.forward(
            linear_out, gamma, beta, bn_param)
        relu_out, relu_cache = ReLU.forward(batchnorm_out)
        cache = (linear_cache, batchnorm_cache, relu_cache)
        return relu_out, cache

    @staticmethod
    def backward(dout, cache):
        linear_cache, batchnorm_cache, relu_cache = cache
        dbatchnorm_out = ReLU.backward(dout, relu_cache)
        dlinear_out, dgamma, dbeta = BatchNorm.backward(dbatchnorm_out, batchnorm_cache)
        dx, dw, db = Linear.backward(dlinear_out, linear_cache)
        return dx, dw, db, dgamma, dbeta

class Conv_BatchNorm_ReLU:
    """Convolution, spatial batch normalization, then a ReLU."""

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        conv_out, conv_cache = FastConv.forward(x, w, b, conv_param)
        sbatchnorm_out, sbatchnorm_cache = SpatialBatchNorm.forward(conv_out, gamma, beta, bn_param)
        relu_out, relu_cache = ReLU.forward(sbatchnorm_out)
        cache = (conv_cache, sbatchnorm_cache, relu_cache)
        return relu_out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, sbatchnorm_cache, relu_cache = cache
        dsbatchnorm_out = ReLU.backward(dout, relu_cache)
        dconv_out, dgamma, dbeta = SpatialBatchNorm.backward(dsbatchnorm_out, sbatchnorm_cache)
        dx, dw, db = FastConv.backward(dconv_out, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool:
    """Convolution, spatial batch normalization, ReLU, then max pooling."""

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        conv_out, conv_cache = FastConv.forward(x, w, b, conv_param)
        sbatchnorm_out, sbatchnorm_cache = SpatialBatchNorm.forward(conv_out, gamma, beta, bn_param)
        relu_out, relu_cache = ReLU.forward(sbatchnorm_out)
        pool_out, pool_cache = FastMaxPool.forward(relu_out, pool_param)
        cache = (conv_cache, sbatchnorm_cache, relu_cache, pool_cache)
        return pool_out, cache


    @staticmethod
    def backward(dout, cache):
        conv_cache, sbatchnorm_cache, relu_cache, pool_cache = cache
        drelu_out = FastMaxPool.backward(dout, pool_cache)
        dsbatchnorm_out = ReLU.backward(drelu_out, relu_cache)
        dconv_out, dgamma, dbeta = SpatialBatchNorm.backward(dsbatchnorm_out, sbatchnorm_cache)
        dx, dw, db = FastConv.backward(dconv_out, conv_cache)
        return dx, dw, db, dgamma, dbeta
