import torch


class ReLU:

    @staticmethod
    def forward(
        x : torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = x.clamp(min=0.0)
        cache = x

        return out, cache

    @staticmethod
    def backward(
        dout    : torch.Tensor,
        cache   : torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        x = cache
        dx = dout * (x > 0)

        return dx
