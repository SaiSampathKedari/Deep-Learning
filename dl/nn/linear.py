import torch


class Linear:

    @staticmethod
    def forward(
        x : torch.Tensor,
        w : torch.Tensor,
        b : torch.Tensor
    )-> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        out = x.reshape(x.shape[0], -1) @ w + b
        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(
        dout    : torch.Tensor,
        cache   : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """

        x, w, b = cache

        dx = (dout @ w.T).reshape(x.shape)

        x_flat = x.reshape(x.shape[0], -1)
        dw = x_flat.T @ dout

        db = dout.sum(dim=0)


        return (dx, dw, db)
