import torch

from dl.nn import Dropout, Linear, Linear_ReLU, SoftmaxCrossEntropy


class FullyConnectedNet:
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dims = 3 * 32 * 32,
        num_classes = 10,
        dropout = 0.0,
        reg = 0.0,
        weight_scale = 1e-2,
        seed=None,
        dtype=torch.float,
        device="cpu",
    ):
        self.use_dropout = dropout != 0.0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype

        # Initialize the parameters of the network, storing all
        # values in the self.params dictionary. Store weights and biases
        # for the first layer in W1 and b1; for the second layer use W2 and
        # b2, etc. Weights should be initialized from a normal distribution
        # centered at 0 with standard deviation equal to weight_scale. Biases
        # should be initialized to zero.

        # dims = [3072, H1, H2, . . . , 10]
        dims = [input_dims, *hidden_dims, num_classes]
        self.params = {}
        for i in range(1, self.num_layers+1):
            self.params[f"W{i}"] = weight_scale * torch.randn(
                dims[i-1],
                dims[i],
                dtype=self.dtype,
                device=device,
            )

            self.params[f"b{i}"] = torch.zeros(
                dims[i],
                dtype=self.dtype,
                device=device,
            )

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.

        self.dropout_param = {}

        if self.use_dropout:
            self.dropout_param= {
                "mode": "train",
                "p": dropout,
            }

        if seed is not None:
            self.dropout_param["seed"] = seed

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
            "num_layers": self.num_layers,
            "use_dropout": self.use_dropout,
            "dropout_param": self.dropout_param,
        }

        torch.save(checkpoint, path)
        print(f"Saved in {path}")

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")

        self.reg = checkpoint["reg"]
        self.dtype = dtype
        self.params = checkpoint["params"]
        self.num_layers = checkpoint["num_layers"]
        self.use_dropout = checkpoint["use_dropout"]
        self.dropout_param = checkpoint["dropout_param"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print(f"load checkpoint file: {path}")

    def _forward(
        self,
        X   : torch.Tensor,
        mode: str = "test",
    ):
        X = X.to(self.dtype)

        if self.use_dropout:
            self.dropout_param['mode'] = mode

        caches = []
        h = X

        # Hidden layers: 1, ..., L - 1
        for i in range(1, self.num_layers):
            h, linear_relu_cache = Linear_ReLU.forward(
                x=h,
                w=self.params[f"W{i}"],
                b=self.params[f"b{i}"],
            )

            if self.use_dropout:
                h, dropout_cache = Dropout.forward(
                    x=h,
                    dropout_param=self.dropout_param,
                )

                # Bundle the two caches for this hidden layer
                layer_cache = (
                    linear_relu_cache,
                    dropout_cache,
                )

            else:
                layer_cache = linear_relu_cache

            caches.append(layer_cache)

        # Final layer: linear only, no ReLU or dropout
        scores, final_cache = Linear.forward(
            x=h,
            w=self.params[f"W{self.num_layers}"],
            b=self.params[f"b{self.num_layers}"],
        )

        caches.append(final_cache)

        return scores, caches

    def _backward(
        self,
        dscores: torch.Tensor,
        caches: list,
    )-> dict[str, torch.Tensor]:

        grads = {}

        # Final linear layer
        final_cache = caches.pop()

        dout, dW, db = Linear.backward(
            dscores,
            final_cache,
        )

        W = self.params[f"W{self.num_layers}"]

        # Store gradients for the final layer
        grads[f"W{self.num_layers}"] = (
            dW + 2.0 * self.reg * W
        )
        grads[f"b{self.num_layers}"] = db

        # Hidden layers: L - 1, L - 2, ..., 1
        for i in range(self.num_layers - 1, 0,-1):
            layer_cache = caches.pop()

            # Reverse dropout before reversing Linear-ReLU
            if self.use_dropout:
                linear_relu_cache, dropout_cache = layer_cache

                dout = Dropout.backward(
                    dout,
                    dropout_cache,
                )
            else:
                linear_relu_cache = layer_cache

            # Reverse ReLU and then Linear
            dout, dW, db = Linear_ReLU.backward(
                dout,
                linear_relu_cache,
            )

            W = self.params[f"W{i}"]

            # Store gradients for this hidden layer
            grads[f"W{i}"] = dW + 2.0 * self.reg * W
            grads[f"b{i}"] = db

        return grads

    def loss(
        self,
        X   : torch.Tensor,
        y   : torch.Tensor | None = None
    ):
        mode = "test" if y is None else "train"

        # forward pass to compute loss, and store cache
        scores, caches = self._forward(X, mode)

        if y is None:
            return scores

        data_loss, dscores = SoftmaxCrossEntropy.forward(
            scores,
            y,
        )

        reg_loss = torch.zeros(
            (),
            dtype=self.dtype,
            device=scores.device,
        )

        for layer_idx in range(1, self.num_layers + 1):
            W = self.params[f"W{layer_idx}"]
            reg_loss += self.reg * torch.sum(W * W)

        loss = data_loss + reg_loss

        # backward pass to compute gradient
        grads = self._backward(dscores, caches)

        return loss, grads

    def predict(
        self,
        X   :   torch.Tensor
    )-> torch.Tensor:

        scores = self.loss(X)
        return scores.argmax(dim=1)
