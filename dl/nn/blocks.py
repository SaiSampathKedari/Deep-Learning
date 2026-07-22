import torch

from .activation import ReLU
from .linear import Linear


class Linear_ReLU:

    @staticmethod
    def forward(
        x   :   torch.Tensor,
        w   :   torch.Tensor,
        b   :   torch.Tensor
    )-> tuple[torch.Tensor, tuple]:
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        fc_out, fc_cache = Linear.forward(x, w, b)
        relu_out, relu_cache  = ReLU.forward(fc_out)

        cache = (fc_cache, relu_cache)
        return relu_out, cache

    @staticmethod
    def backward(
        dout    : torch.Tensor,
        cache   : tuple
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache

        fc_dout = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(fc_dout, fc_cache)

        return (dx, dw, db)
