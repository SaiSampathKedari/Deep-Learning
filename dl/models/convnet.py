import torch
from typing import Literal, Sequence
from dl.nn.init import kaiming_initializer, gaussian_initializer
from dl.nn.blocks import Conv_BatchNorm_ReLU_Pool, Conv_BatchNorm_ReLU
from dl.nn.blocks import Conv_ReLU, Conv_ReLU_Pool
from dl.nn import Linear
from dl.nn.loss import SoftmaxCrossEntropy


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    # Fixed architecture of every convolution
    KERNEL_SIZE = 3
    CONV_STRIDE = 1
    CONV_PADDING = (KERNEL_SIZE - 1) // 2

    # Fixed architecture of every pooling layer
    POOL_SIZE = 2
    POOL_STRIDE = 2
    
    def __init__(
        self,
        input_dims: tuple[int, int, int] = (3, 32, 32),
        num_filters: Sequence[int] = (8, 8, 8, 8, 8),
        max_pools: Sequence[int] = (0, 1, 2, 3, 4),
        batchnorm: bool = False,
        num_classes: int = 10,
        weight_init: Literal["kaiming", "gaussian"] = "kaiming",
        weight_scale: float = 1e-3,
        reg: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Manual VGG-style convolutional network.

        Architecture:
            {conv - [batchnorm] - relu - [pool]} x (L - 1) - linear
        """
        
        # Model architecture
        self.input_dims = input_dims
        self.num_filters = tuple(num_filters)
        self.max_pools = frozenset(max_pools)
        self.batchnorm = batchnorm
        self.num_classes = num_classes
        self.num_layers = len(num_filters) + 1
        
        # Initialization and computation configuration
        self.weight_init = weight_init
        self.weight_scale = weight_scale
        self.reg = reg
        self.dtype = dtype
        self.device = torch.device(device)

        # Exact configurations passed to convolution and pooling layers
        self.conv_param: dict[str, int] = {
            "stride": self.CONV_STRIDE,
            "pad": self.CONV_PADDING,
        }

        self.pool_param: dict[str, int] = {
            "pool_height": self.POOL_SIZE,
            "pool_width": self.POOL_SIZE,
            "stride": self.POOL_STRIDE,
        }

        # Learnable parameters
        self.params: dict[str, torch.Tensor] = {}
    
        # Initilizes all the parameters
        self._initialize_parameters()
        
        # Nonlearnable batch-normalization state
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{"mode": "train"} for _ in range(len(num_filters))]
    
    def _initialize_parameters(self) -> None:
        """
        Initialize convolutional and linear parameters while tracking
        the spatial dimensions through the network.
        """
        in_channels, height, width = self.input_dims

        for i in range(len(self.num_filters)):
            layer_index = i + 1
            out_channels = self.num_filters[i]
            
            # Convolution weights and bias's
            if self.weight_init == "kaiming":
                weight = kaiming_initializer(
                    Din=in_channels,
                    Dout=out_channels,
                    kernel_size=self.KERNEL_SIZE,
                    relu=True,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                weight = gaussian_initializer(
                    Din=in_channels,
                    Dout=out_channels,
                    kernel_size=self.KERNEL_SIZE,
                    std=self.weight_scale,
                    dtype=self.dtype,
                    device=self.device,
                )

            bias = torch.zeros(
                out_channels,
                dtype=self.dtype,
                device=self.device,
            )

            self.params[f"W{layer_index}"] = weight
            self.params[f"b{layer_index}"] = bias

            # Spatial batch-normalization parameters
            if self.batchnorm:
                self.params[f"gamma{layer_index}"] = torch.ones(
                    out_channels,
                    dtype=self.dtype,
                    device=self.device,
                )

                self.params[f"beta{layer_index}"] = torch.zeros(
                    out_channels,
                    dtype=self.dtype,
                    device=self.device,
                )

            # Spatial size after convolution
            height = 1 + ( height + 2 * self.CONV_PADDING - self.KERNEL_SIZE) // self.CONV_STRIDE
            width = 1 + ( width + 2 * self.CONV_PADDING - self.KERNEL_SIZE ) // self.CONV_STRIDE

            # Spatial size after optional max pooling
            if i in self.max_pools:
                height = 1 + (height - self.POOL_SIZE) // self.POOL_STRIDE
                width = 1 + (width - self.POOL_SIZE) // self.POOL_STRIDE

            # Current output channels become next layer's input channels
            in_channels = out_channels

        # Final linear layer
        final_layer = self.num_layers
        flattened_dim = in_channels * height * width

        if self.weight_init == "kaiming":
            final_weight = kaiming_initializer(
                Din=flattened_dim,
                Dout=self.num_classes,
                relu=False,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            final_weight = gaussian_initializer(
                Din=flattened_dim,
                Dout=self.num_classes,
                std=self.weight_scale,
                dtype=self.dtype,
                device=self.device,
            )

        final_bias = torch.zeros(
            self.num_classes,
            dtype=self.dtype,
            device=self.device,
        )

        self.params[f"W{final_layer}"] = final_weight
        self.params[f"b{final_layer}"] = final_bias


    def save(self, path: str) -> None:
        """
        Save the complete model state and architecture configuration.
        """
        checkpoint = {
            "format_version": 1,

            # Architecture
            "input_dims": self.input_dims,
            "num_filters": self.num_filters,
            "max_pools": tuple(self.max_pools),
            "batchnorm": self.batchnorm,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,

            # Layer configuration
            "conv_param": self.conv_param,
            "pool_param": self.pool_param,

            # Initialization and regularization configuration
            "weight_init": self.weight_init,
            "weight_scale": self.weight_scale,
            "reg": self.reg,

            # Learnable parameters and nonlearnable state
            "params": self.params,
            "bn_params": self.bn_params,
        }
        torch.save(checkpoint, path)
    
    def load(
        self,
        path: str,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Load model parameters, architecture configuration, and batchnorm state.

        dtype and device optionally specify where the loaded tensors should be
        placed. If omitted, the model's current dtype and device are used.
        """
        target_dtype = self.dtype if dtype is None else dtype
        target_device = (
            self.device
            if device is None
            else torch.device(device)
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        # Architecture
        self.input_dims = tuple(checkpoint["input_dims"])
        self.num_filters = tuple(checkpoint["num_filters"])
        self.max_pools = frozenset(checkpoint["max_pools"])
        self.batchnorm = checkpoint["batchnorm"]
        self.num_classes = checkpoint["num_classes"]
        self.num_layers = checkpoint["num_layers"]

        # Layer configuration
        self.conv_param = checkpoint["conv_param"]
        self.pool_param = checkpoint["pool_param"]

        # Initialization and regularization configuration
        self.weight_init = checkpoint["weight_init"]
        self.weight_scale = checkpoint["weight_scale"]
        self.reg = checkpoint["reg"]

        # Runtime configuration
        self.dtype = target_dtype
        self.device = target_device

        # Move all learnable parameters
        self.params = {
            name: parameter.to(
                device=self.device,
                dtype=self.dtype,
            )
            for name, parameter in checkpoint["params"].items()
        }

        # Move all tensor-valued batchnorm state
        self.bn_params = checkpoint["bn_params"]

        for bn_param in self.bn_params:
            for name, value in bn_param.items():
                if isinstance(value, torch.Tensor):
                    bn_param[name] = value.to(
                        device=self.device,
                        dtype=self.dtype,
                    )
    
    
    def _forward(
        self,
        X: torch.Tensor,
        mode: str = "test",
    ) -> tuple[torch.Tensor, tuple[list, tuple]]:
        """
        Compute the forward pass.

        Returns:
        - scores: Class scores of shape (N, num_classes)
        - caches: Tuple containing:
            - macro_caches: One cache for each convolutional macro-layer
            - final_cache: Cache for the final linear layer
        """
        X = X.to(
            device=self.device,
            dtype=self.dtype,
        )

        # Batch normalization behaves differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        h = X
        macro_caches = []

        # Convolutional macro-layers: 1, ..., L - 1
        for layer_index in range(1, self.num_layers):
            W = self.params[f"W{layer_index}"]
            b = self.params[f"b{layer_index}"]

            # max_pools uses zero-based macro-layer indices.
            macro_index = layer_index - 1
            use_pool = macro_index in self.max_pools

            if self.batchnorm:
                gamma = self.params[f"gamma{layer_index}"]
                beta = self.params[f"beta{layer_index}"]
                bn_param = self.bn_params[macro_index]

                if use_pool:
                    h, cache = Conv_BatchNorm_ReLU_Pool.forward(
                        x=h,
                        w=W,
                        b=b,
                        gamma=gamma,
                        beta=beta,
                        conv_param=self.conv_param,
                        bn_param=bn_param,
                        pool_param=self.pool_param,
                    )
                else:
                    h, cache = Conv_BatchNorm_ReLU.forward(
                        x=h,
                        w=W,
                        b=b,
                        gamma=gamma,
                        beta=beta,
                        conv_param=self.conv_param,
                        bn_param=bn_param,
                    )

            else:
                if use_pool:
                    h, cache = Conv_ReLU_Pool.forward(
                        x=h,
                        w=W,
                        b=b,
                        conv_param=self.conv_param,
                        pool_param=self.pool_param,
                    )
                else:
                    h, cache = Conv_ReLU.forward(
                        x=h,
                        w=W,
                        b=b,
                        conv_param=self.conv_param,
                    )

            macro_caches.append(cache)

        # Final layer: linear only
        final_layer = self.num_layers

        scores, final_cache = Linear.forward(
            x=h,
            w=self.params[f"W{final_layer}"],
            b=self.params[f"b{final_layer}"],
        )

        caches = (macro_caches, final_cache)

        return scores, caches
        
    
    def _backward(
        self,
        dscores: torch.Tensor,
        caches: tuple[list, tuple],
    ) -> dict[str, torch.Tensor]:
        """
        Compute gradients of all learnable parameters.
        """
        macro_caches, final_cache = caches

        grads: dict[str, torch.Tensor] = {}

        # Final linear layer
        final_layer = self.num_layers

        dout, dW, db = Linear.backward(dscores, final_cache)

        W = self.params[f"W{final_layer}"]

        grads[f"W{final_layer}"] = dW + 2.0 * self.reg * W
        grads[f"b{final_layer}"] = db

        # Convolutional macro-layers:
        # L - 1, L - 2, ..., 1
        for layer_index in range(self.num_layers - 1, 0, -1):
            macro_index = layer_index - 1
            cache = macro_caches[macro_index]

            use_pool = macro_index in self.max_pools

            if self.batchnorm:
                if use_pool:
                    dout, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, cache)
                else:
                    dout, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, cache)

                grads[f"gamma{layer_index}"] = dgamma
                grads[f"beta{layer_index}"] = dbeta

            else:
                if use_pool:
                    dout, dW, db = Conv_ReLU_Pool.backward(dout, cache)
                else:
                    dout, dW, db = Conv_ReLU.backward(dout, cache)

            W = self.params[f"W{layer_index}"]

            grads[f"W{layer_index}"] = dW + 2.0 * self.reg * W
            grads[f"b{layer_index}"] = db

        return grads
    
    def loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
    ):
        """
        Compute class scores during testing, or loss and gradients during training.
        """
        mode = "test" if y is None else "train"

        scores, caches = self._forward(X, mode=mode)

        if y is None:
            return scores

        y = y.to(device=self.device)

        data_loss, dscores = SoftmaxCrossEntropy.forward(scores, y)

        reg_loss = torch.zeros((), dtype=self.dtype, device=self.device)

        # Regularize convolutional weights and final linear weights.
        for layer_index in range(1, self.num_layers + 1):
            W = self.params[f"W{layer_index}"]
            reg_loss += self.reg * torch.sum(W * W)

        loss = data_loss + reg_loss
        grads = self._backward( dscores, caches)
        return loss, grads
    
    def predict(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the class index for each input sample.
        """
        scores = self.loss(X)

        return scores.argmax(dim=1)