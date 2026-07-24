import torch


class BatchNorm(object):

    @staticmethod
    def forward(
        x       :   torch.Tensor,
        gamma   :   torch.Tensor,
        beta    :   torch.Tensor,
        bn_param:   dict,
    ):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            sample_mean = x.mean(dim=0)
            x_centered = x - sample_mean

            sample_var = torch.mean(
                torch.square(x_centered),
                dim=0,
            )

            inv_std = torch.rsqrt(sample_var + eps)
            x_hat = x_centered * inv_std

            out = gamma * x_hat + beta

            running_mean = (
                momentum * running_mean
                + (1.0 - momentum) * sample_mean
            )

            running_var = (
                momentum * running_var
                + (1.0 - momentum) * sample_var
            )

            cache = (mode, x_hat, gamma, inv_std)
        elif mode == 'test':
            inv_std = torch.rsqrt(running_var + eps)
            x_hat = (x - running_mean) * inv_std
            out = gamma * x_hat + beta
            cache = (mode, x_hat, gamma, inv_std)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(
        dout    :   torch.Tensor,
        cache   :   tuple    
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dx, dgamma, dbeta = None, None, None
        mode, x_hat, gamma, inv_std = cache
        N, _ = dout.shape

        dgamma = (dout * x_hat).sum(dim=0)
        dbeta = dout.sum(dim=0)

        dx_hat = dout * gamma

        if mode == "train":
            dx = (
                inv_std
                / N
                * (
                    N * dx_hat
                    - dx_hat.sum(dim=0, keepdim=True)
                    - x_hat
                    * (dx_hat * x_hat).sum(
                        dim=0,
                        keepdim=True,
                    )
                )
            )
        elif mode == "test":
            # Running statistics are constants during testing.
            dx = dx_hat * inv_std

        else:
            raise ValueError(
                f'Invalid batch-normalization mode: "{mode}".'
            )

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        raise NotImplementedError


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        out, cache = None, None
        N, C, H, W = x.shape

        x_flat = (
            x.permute(0, 2, 3, 1)
            .reshape(N * H * W, C)
        )

        out_flat, cache = BatchNorm.forward(
            x_flat,
            gamma,
            beta,
            bn_param,
        )

        out = (
            out_flat.reshape(N, H, W, C)
            .permute(0, 3, 1, 2)
        )
        return out, cache

    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = None, None, None
        N, C, H, W = dout.shape

        dout_flat = (
            dout.permute(0, 2, 3, 1)
            .reshape(N * H * W, C)
        )

        dx_flat, dgamma, dbeta = BatchNorm.backward(
            dout_flat,
            cache,
        )

        dx = (
            dx_flat.reshape(N, H, W, C)
            .permute(0, 3, 1, 2)
        )

        return dx, dgamma, dbeta