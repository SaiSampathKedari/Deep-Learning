import torch


class Conv:

    @staticmethod
    def forward(
        x           :   torch.Tensor,
        w           :   torch.Tensor, 
        b           :   torch.Tensor, 
        conv_param  :   dict[str, int]
    ):
        """
        A naive implementation of the forward pass for a convolutional layer.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        pad = conv_param["pad"]
        stride = conv_param["stride"]

        N, C, H_in, W_in = x.shape
        F, C_w, HH, WW = w.shape

        if C_w != C:
            raise ValueError(
                f"Input has {C} channels, but filters expect {C_w}."
            )

        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        H_out = 1 + (H_in + 2 * pad - HH) // stride
        W_out = 1 + (W_in + 2 * pad - WW) // stride

        out = torch.zeros(
            (N, F, H_out, W_out),
            dtype=x.dtype,
            device=x.device,
        )

        for i in range(H_out):
            for j in range(W_out):
                h0 = i * stride
                w0 = j * stride
                h1 = h0 + HH
                w1 = w0 + WW

                window = x_padded[:, :, h0:h1, w0:w1]

                product = (
                    window.reshape(N, 1, C, HH, WW)
                    * w.reshape(1, F, C, HH, WW)
                )

                out[:, :, i, j] = (
                    product.sum(dim=(2, 3, 4))
                    + b.reshape(1, F)
                )

        cache = (x, w, b, conv_param)

        return out, cache

    @staticmethod
    def backward(
      dout  : torch.Tensor, 
      cache : tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]
    ):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        x, w, _, conv_param = cache
        
        pad = conv_param["pad"]
        stride = conv_param["stride"]
                
        N, C, _, _ = x.shape
        F, _, HH, WW = w.shape
        _, _, H_out, W_out = dout.shape

        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        
        dx_padded = torch.zeros_like(x_padded)
        dw = torch.zeros_like(w)
        db = dout.sum(dim=(0, 2, 3)) #(F, )
        
        for i in range(H_out):
          for j in range(W_out):
              h0 = i * stride
              w0 = j * stride
              h1 = h0 + HH
              w1 = w0 + WW
              
              window = x_padded[:, :, h0:h1, w0:w1] # (N, C, HH, WW)
              dout_ij = dout[:, :, i, j] # (N, F)
              
              # Total: 2 input's window, w. 1 output dout_ij
              # window   = (N, C, HH, WW)
              # w        = (F, C, HH, WW) 
              # dout_ij  = (N, F)
              
              dw += (
                dout_ij.reshape(N, F, 1, 1, 1) 
                * window.reshape(N, 1, C, HH, WW)
              ).sum(dim=0)
              
              dx_padded[:, :, h0:h1, w0:w1] += (
                dout_ij.reshape(N, F, 1, 1, 1) 
                * w.reshape(1, F, C, HH, WW)
              ).sum(dim=1)
        
        if pad == 0:
          dx = dx_padded
        else:
          dx = dx_padded[:, :, pad:-pad, pad:-pad]
        
        return dx, dw, db

class FastConv:
    """Reference implementation backed by ``torch.nn.Conv2d``.

    Used to check the naive ``Conv`` above and to keep deep models trainable
    in reasonable time.
    """

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db
