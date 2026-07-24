import torch


class MaxPool:

    @staticmethod
    def forward(
        x           :   torch.Tensor, 
        pool_param  :   dict[str, int]
    ):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        N, C, H_in, W_in = x.shape
        
        pool_height = pool_param["pool_height"]
        pool_width = pool_param["pool_width"]
        stride = pool_param["stride"]
        
        H_out = 1 + (H_in - pool_height)//stride
        W_out = 1 + (W_in - pool_width)//stride
        
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        for i in range(H_out):
            for j in range(W_out):
                h0 = i * stride
                w0 = j * stride
                h1 = h0 + pool_height
                w1 = w0 + pool_width
                
                window = x[:, :, h0:h1, w0:w1]
                out[:, :, i, j] = window.amax(dim=(2, 3))
        
        cache = (x, pool_param)
        
        return out, cache

    @staticmethod
    def backward(
        dout    :   torch.Tensor,
        cache   :   tuple[torch.Tensor, dict[str, int]]
    ):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        
        Returns:
        - dx: Gradient with respect to x
        """
        x, pool_param = cache
        N, C, H_out, W_out = dout.shape
        
        pool_height = pool_param["pool_height"]
        pool_width = pool_param["pool_width"]
        stride = pool_param["stride"]
        
        dx = torch.zeros_like(x)
        
        for i in range(H_out):
            for j in range(W_out):
                h0 = i * stride
                w0 = j * stride
                h1 = h0 + pool_height
                w1 = w0 + pool_width
                
                window = x[:, :, h0:h1, w0:w1]
                mask = (window  == window.amax(dim=(2, 3), keepdim=True))
                
                dx[:, :, h0:h1, w0:w1] += mask * dout[:, :, i, j].reshape(N, C, 1, 1)
        
        return dx

class FastMaxPool:
    """Reference implementation backed by ``torch.nn.MaxPool2d``."""

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx
