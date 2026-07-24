import torch


def kaiming_initializer(
    Din     : int,
    Dout    : int,
    K       : int|None=None,
    relu    : bool=True,
    device="cpu",
    dtype=torch.float32,
):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        fan_in = Din
        weight_scale = (gain / fan_in) ** 0.5
        weight = weight_scale * torch.randn((Din, Dout), dtype=dtype, device=device)
    else:
        fan_in = Din * K * K
        weight_scale = (gain / fan_in) ** 0.5
        weight = weight_scale * torch.randn((Dout, Din, K, K), dtype=dtype, device= device)
    return weight