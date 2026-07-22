"""Parameter update rules, mirroring ``torch.optim``.

Each rule takes ``(W, dW, config)`` and returns ``(next_W, config)``; the
config dict carries any per-parameter state (velocity, cache, moments).
"""
from typing import Any

import torch


def sgd(
    W       :   torch.Tensor,
    dW      :   torch.Tensor,
    config  :   dict[str, Any] | None = None,
)-> tuple[torch.Tensor, dict[str, Any]]:
    """
    Perform one vanilla stochastic gradient descent update.

    config format:
        - learning_rate: Scalar learning rate..
    """

    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    if W.shape != dW.shape:
        raise ValueError(
            f"`W` and `dW` must have the same shape, "
            f"but received {W.shape} and {dW.shape}."
        )

    next_W = W - config['learning_rate'] * dW

    return next_W, config

def sgd_momentum(
    W       :   torch.Tensor,
    dW      :   torch.Tensor,
    config  :    dict[str, Any] | None = None,
    )-> tuple[torch.Tensor, dict[str, Any]]:
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    config.setdefault("velocity", torch.zeros_like(W))

    learning_rate = config["learning_rate"]
    momentum = config["momentum"]
    velocity = config["velocity"]

    velocity = momentum * velocity + dW
    next_W = W - learning_rate * velocity

    config["velocity"] = velocity

    return next_W, config

def rmsprop(
    W: torch.Tensor,
    dW: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Update a parameter using RMSProp.

    Inputs:
    - W: Parameter tensor.
    - dW: Gradient with respect to W.
    - config: Optimizer settings and squared-gradient state.

    Returns:
    - Updated W and config.
    """
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", torch.zeros_like(W))

    learning_rate = config["learning_rate"]
    decay_rate = config["decay_rate"]
    epsilon = config["epsilon"]
    cache = config["cache"]

    grad_squared = decay_rate * cache + (1.0 - decay_rate) * torch.square(dW)

    next_W = W - learning_rate * dW / (
        torch.sqrt(grad_squared) + epsilon
    )

    config["cache"] = grad_squared

    return next_W, config

def adam(
    W: torch.Tensor,
    dW: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Update a parameter using Adam.

    Inputs:
    - W: Parameter tensor.
    - dW: Gradient with respect to W.
    - config: Optimizer settings and moment estimates.

    Returns:
    - Updated W and config.
    """
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", torch.zeros_like(W))
    config.setdefault("v", torch.zeros_like(W))
    config.setdefault("t", 0)

    learning_rate = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    epsilon = config["epsilon"]
    m = config["m"]
    v = config["v"]
    t = config["t"] + 1

    # Moving averages
    m = beta1 * m + (1.0 - beta1) * dW
    v = beta2 * v + (1.0 - beta2) * torch.square(dW)

    # Bias correction
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)

    # Parameter update
    next_W = W - learning_rate * m_hat / (
        torch.sqrt(v_hat) + epsilon
    )

    config["m"] = m
    config["v"] = v
    config["t"] = t

    return next_W, config
