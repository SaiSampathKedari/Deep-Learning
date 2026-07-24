import torch


def kaiming_initializer(
    Din         : int,
    Dout        : int,
    kernel_size : int | None = None,
    relu        : bool = True,
    device      : str | torch.device = "cpu",
    dtype       : torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Initialize linear or convolutional weights using Kaiming scaling.

    Weights are drawn from a Gaussian with standard deviation
    sqrt(gain / fan_in), which keeps the activation variance roughly constant
    as depth increases. Without it, a deep stack of layers drives activations
    (and therefore gradients) toward zero.

    Inputs:
    - Din: Number of input dimensions (linear) or input channels (conv).
    - Dout: Number of output dimensions (linear) or output channels (conv).
    - kernel_size: If None, initialize a linear weight of shape (Din, Dout);
      otherwise a convolutional weight of shape
      (Dout, Din, kernel_size, kernel_size).
    - relu: If True use a gain of 2, appropriate for a layer followed by a
      ReLU; otherwise use a gain of 1.
    - device, dtype: Placement and precision of the returned tensor.

    Returns:
    - weight: Initialized weight tensor.
    """
    gain = 2.0 if relu else 1.0

    if kernel_size is None:
        shape = (Din, Dout)
        fan_in = Din
    else:
        shape = (Dout, Din, kernel_size, kernel_size)
        fan_in = Din * kernel_size * kernel_size

    weight_scale = (gain / fan_in) ** 0.5

    return weight_scale * torch.randn(shape, dtype=dtype, device=device)


def gaussian_initializer(
    Din         : int,
    Dout        : int,
    kernel_size : int | None = None,
    std         : float = 1e-3,
    device      : str | torch.device = "cpu",
    dtype       : torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Initialize linear or convolutional weights from a zero-mean Gaussian with
    a fixed standard deviation.

    The scale is independent of layer size, so it does not preserve activation
    variance through depth; prefer kaiming_initializer for deep networks.

    Inputs:
    - Din: Number of input dimensions (linear) or input channels (conv).
    - Dout: Number of output dimensions (linear) or output channels (conv).
    - kernel_size: If None, initialize a linear weight of shape (Din, Dout);
      otherwise a convolutional weight of shape
      (Dout, Din, kernel_size, kernel_size).
    - std: Standard deviation of the Gaussian.
    - device, dtype: Placement and precision of the returned tensor.

    Returns:
    - weight: Initialized weight tensor.
    """
    if kernel_size is None:
        shape = (Din, Dout)
    else:
        shape = (Dout, Din, kernel_size, kernel_size)

    return std * torch.randn(shape, dtype=dtype, device=device)
