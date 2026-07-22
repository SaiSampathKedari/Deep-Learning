import torch


class Dropout:

    @staticmethod
    def forward(
        x   : torch.Tensor,
        dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
        - p: Dropout parameter. We *drop* each neuron output with
            probability p.
        - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
        if the mode is test, then just return the input.
        - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
        is the dropout mask that was used to multiply the input; in
        test mode, mask is None.
        """
        p = dropout_param["p"]
        mode = dropout_param["mode"]

        if not 0 <= p < 1.0:
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}")

        if "seed" in dropout_param:
            torch.manual_seed(dropout_param["seed"])

        mask = None
        out = None
        if mode == "train":
            mask = (torch.rand_like(x) >= p).to(x.dtype) / (1.0 - p)
            out = x * mask
        elif mode == "test":
            out = x
        else:
            raise ValueError(f"Invalid dropout mode : {mode}")

        cache = (dropout_param, mask)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """

        dropout_param, mask = cache
        mode = dropout_param["mode"]

        dx = None
        if mode == "train":
            dx = dout * mask
        elif mode == "test":
            dx = dout
        else:
            raise ValueError(f"Invalid dropout mode : {mode}")

        return dx
