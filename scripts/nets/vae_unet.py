"""Code from the Improved Denoising Diffusion Models codebase:
https://github.com/openai/improved-diffusion"""

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import logging


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    # logging.info(kwargs)
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0,
                                          end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat(
            [embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True)
                             for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads



class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
        padding_mode: str = 'zeros'
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(
            1, embed_dim, 3 * embed_dim, 1, padding_mode=padding_mode)
        self.c_proj = conv_nd(
            1, embed_dim, output_dim or embed_dim, 1, padding_mode=padding_mode)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x):
        assert x.shape[1] == self.channels, f"{x.shape[1]} != {self.channels}"
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding_mode='zeros'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1, padding_mode=padding_mode
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        padding_mode='zeros'
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3,
                    padding=1, padding_mode=padding_mode),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels,
                        3, padding=1, padding_mode=padding_mode)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, padding_mode=padding_mode
            )
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1, padding_mode=padding_mode)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        padding_mode='zeros'
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3,
                           1, padding_mode=padding_mode)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(
            conv_nd(1, channels, channels, 1, padding_mode=padding_mode))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3,
                              length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)



def vae_encoding(vae_encoder, latent_dim, xt, emb, cond = None, prior = None):

    """
    Function to modularize the VAE Noise injection process, independent of where the injection is happening
    """

    kl_div = None

    # In Training mode, we have the conditioning signal
    if cond is not None:

        # VAE Encoder
        mu, logvar = vae_encoder(xt, cond, emb)

        # Upscale to image size if latent dim is channels * image size * image size / 16 (for bold injections)
        if latent_dim == int(xt.shape[1] * xt.shape[2] * xt.shape[3]/16):
            mu = mu.reshape(mu.shape[0], xt.shape[1], int(xt.shape[2]/4), int(xt.shape[3]/4))
            logvar = logvar.reshape(logvar.shape[0], xt.shape[1], int(xt.shape[2]/4), int(xt.shape[3]/4))
            mu = th.repeat_interleave(mu, 4, dim=2)
            mu = th.repeat_interleave(mu, 4, dim=3)
            logvar = th.repeat_interleave(logvar, 4, dim=2)
            logvar = th.repeat_interleave(logvar, 4, dim=3)

        # Reparameterization trick
        z_sample = th.randn_like(logvar) * th.exp(0.5*logvar) + mu

        # KL Divergence for VAE Encoder
        kl_div = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean()

    # In Generation mode, we don't have the conditioning signal
    else:
        if prior is None:
            if latent_dim == xt.shape[1] * xt.shape[2] * xt.shape[3] / 16:
                z_sample = th.randn(xt.shape[0], latent_dim*16).to(xt.device)
            else:
                z_sample = th.randn(xt.shape[0], latent_dim).to(xt.device) # Batch dim * latent dim
        else:
            z_sample = prior.to(xt.device)

    return z_sample, kl_div

            

class VAEEncoder(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        dim,
        num_res_blocks,
        attention_levels,
        dropout,
        ch_mult,
        embed_dim,
        latent_dim,
        bottleneck=False,
        use_checkpoint=False
    ):
        super().__init__()

        # Arguments to set up the model
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = dim
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels # (2,3) or (2,)
        self.dropout = dropout 
        self.channel_mult = ch_mult
        time_embed_dim = embed_dim
        self.bottleneck = bottleneck

        # Default Arguments
        self.conv_resample = True
        self.num_classes = None
        self.use_checkpoint = False
        self.dtype = th.float32 # Default to float32, check that this doesn't cause issues
        self.num_heads = 1
        self.num_head_channels = -1
        self.num_heads_upsample = self.num_heads
        self.use_new_attention_order = True
        dims = 2  # Constant for our purposes
        self.use_scale_shift_norm = False
        self.resblock_updown = False
        self.padding_mode = 'zeros'


        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels*2, ch, 3, padding=1,
                                             padding_mode=self.padding_mode))]
        )
        curr_res = self.image_size
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 0  # Counter for selecting the levels where attention is applied
        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        padding_mode=self.padding_mode
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            padding_mode=self.padding_mode
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            padding_mode=self.padding_mode
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=dims, out_channels=out_ch,
                            padding_mode=self.padding_mode
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds += 1
                self._feature_size += ch
                curr_res = curr_res // 2

        print("Current Resolution: ", curr_res)
        
        # dense layers for mean and logvar
        self.dense_mean = nn.Linear(input_block_chans[-1]*curr_res*curr_res, latent_dim)
        self.dense_logvar = nn.Linear(input_block_chans[-1]*curr_res*curr_res, latent_dim)



    def forward(self, xt, cond, emb, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Combine the two images
        x = th.cat([xt, cond], dim=1)

        hs = []

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)


        # dense layers for mean and logvar
        h = h.view(h.size(0), -1)
        mean = self.dense_mean(h)
        logvar = self.dense_logvar(h)

        return mean, logvar
    


class VAEUnet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        dim,
        num_res_blocks,
        attention_levels,
        dropout,
        ch_mult,
        latent_dim,
        vae_loc,
        vae_inject,
        var_timestep,
        add_noise = False,
        noise_scale=0.01,
        xt_dropout=0,
        use_checkpoint=False
    ):
        super().__init__()

        # Arguments to set up the model
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = dim
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels # (2,3) or (2,)
        self.dropout = dropout 
        self.channel_mult = ch_mult
        self.time_embed_dim = dim * 4
        self.latent_dim = latent_dim if not vae_loc == 'bold' else int(in_channels * image_size * image_size / 16)
        self.xt_dropout = xt_dropout
        self.var_timestep = var_timestep

        assert vae_loc in ['start', 'bottleneck', 'emb', 'maps', 'bold'], 'VAE location must be one of [start, bottleneck, emb, maps]'
        assert vae_inject in ['concat', 'add'], 'VAE injection must be one of [concat, add]'
        self.vae_loc = vae_loc
        self.vae_inject = vae_inject

        # Default Arguments
        self.conv_resample = True
        self.num_classes = None
        self.use_checkpoint = False
        self.dtype = th.float32 # Default to float32, check that this doesn't cause issues
        self.num_heads = 1
        self.num_head_channels = -1
        self.num_heads_upsample = self.num_heads
        self.use_new_attention_order = True
        self.dims = 2  # Constant for our purposes
        self.use_scale_shift_norm = False
        self.resblock_updown = False
        self.padding_mode = 'zeros'
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.sample_noise = None
        self.counter = 0
        self.max_noise_counter = 50000

        self.time_embed = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        #Xt Dropout Layer
        self.xt_dropout_layer = nn.Dropout(xt_dropout)

        # Variable Timestep Embedding Adjustment
        if self.var_timestep:
            self.time_embed_dim = self.time_embed_dim * 2 # Double the time embedding dimension to include t2 embedding

        # VAE Encoder
        self.vae_encoder = VAEEncoder(image_size, 
                                      in_channels, 
                                      int(dim / 2), # Half dim for VAE Encoder
                                      num_res_blocks, 
                                      attention_levels, 
                                      dropout, 
                                      [int((i+1)/2) for i in ch_mult], # Half channel mult for VAE Encoder
                                      self.time_embed_dim,
                                      self.latent_dim)
        
        # VAE Projection
        self.setup_vae_projection()

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)

        # For bold injections, we need to scale the VAE and sample Noise
        if self.vae_loc == 'bold':
            self.inject_scale = nn.Parameter(th.ones(1)*0.01)

        # Adjust the input channels if VAE is concateded at the start
        if self.vae_loc in ['start','bold'] and self.vae_inject == 'concat':
            #in_channels += self.latent_dim
            in_channels = in_channels*2 # Trying out the projection 
        
        # Adjust the time embedding if VAE is concateded at the start
        if self.vae_loc == 'emb' and self.vae_inject == 'concat':
            self.time_embed_dim = self.time_embed_dim*2 # self.time_embed_dim

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(self.dims, in_channels, ch, 3, padding=1,
                                             padding_mode=self.padding_mode))]
        )


        curr_res = self.image_size
        
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 0  # Counter for selecting the levels where attention is applied
        for level, mult in enumerate(self.channel_mult):
        
            for _ in range(num_res_blocks):

                if self.vae_loc == 'maps' and self.vae_inject == 'concat':
                    ch = ch*2

                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        padding_mode=self.padding_mode
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            padding_mode=self.padding_mode
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1:
                out_ch = ch

                if self.vae_loc == 'maps' and self.vae_inject == 'concat':
                    ch = ch*2

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            padding_mode=self.padding_mode
                        )
                        if self.resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self.dims, out_channels=out_ch,
                            padding_mode=self.padding_mode
                        )
                    )
                )

                ch = out_ch
                input_block_chans.append(ch)
                ds += 1
                self._feature_size += ch

                curr_res = curr_res // 2


        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=self.dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                padding_mode=self.padding_mode
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
                padding_mode=self.padding_mode
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                dims=self.dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                padding_mode=self.padding_mode
            ),
        )
        self._feature_size += ch

        if self.vae_loc == 'bottleneck' and self.vae_inject == 'concat':
            # ch += self.latent_dim
            ch = ch*2

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
        
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()

                # if self.vae_loc == 'maps' and self.vae_inject == 'concat':
                #     ch = ch*2

                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        padding_mode=self.padding_mode
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in attention_levels:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                            padding_mode=self.padding_mode
                        )
                    )


                if level and i == num_res_blocks:
                    out_ch = ch

                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                            padding_mode=self.padding_mode
                        )
                        if self.resblock_updown
                        else Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch,
                                      padding_mode=self.padding_mode)
                    )
                    ds -= 1

                    curr_res = curr_res * 2



                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch


        ch = input_ch
        # if self.vae_loc == 'maps' and self.vae_inject == 'concat':
        #     ch = ch*2

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, ch, self.out_channels, 3,
                        padding=1, padding_mode=self.padding_mode)),
        )


    def setup_vae_projection(self):

        # VAE Projection for non-map-wise injection
        if not self.vae_loc in ['maps', 'bold']: # Different treatment for when injection happens at feature maps
            if self.vae_loc == 'start':
                target_dim = self.in_channels * self.image_size * self.image_size
            elif self.vae_loc == 'bottleneck':
                bottleneck_ch = int(self.channel_mult[-1]*self.model_channels) 
                feature_dim = int(self.image_size / 2**(len(self.channel_mult)-1))
                target_dim = bottleneck_ch * feature_dim * feature_dim        
            elif self.vae_loc == 'emb':
                target_dim = self.time_embed_dim # *2 # Double the time embedding dimension for double the sauce
                #self.time_embed_dim = self.time_embed_dim + target_dim # Increase the time embedding dimension to be able to concatenate with VAE latent
            else:
                raise ValueError('VAE Injection point invalid')
            
            # Project latent from latent dim to U-Net Dim, depending on injection point
            # Multiple linear layers as projection
            self.vae_projection = []
            for i in range(5):
                self.vae_projection.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.vae_projection.append(nn.Linear(self.latent_dim, target_dim))
            self.vae_projection = nn.Sequential(*self.vae_projection)

            print("VAE Projection with target dim: ", target_dim)



        # VAE Projection for map-wise injection
        curr_res = self.image_size
        self.vae_projections = nn.ModuleList([])

        # # VAE Projection for xt
        # proj = []
        # for i in range(5):
        #     proj.append(nn.Linear(self.latent_dim, self.latent_dim))
        # proj.append(nn.Linear(self.latent_dim, self.in_channels*curr_res*curr_res))
        # proj = nn.Sequential(*proj)
        # self.vae_projections.append(proj)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)

        self._feature_size = ch
        ds = 0  # Counter for selecting the levels where attention is applied
        for level, mult in enumerate(self.channel_mult):

            for _ in range(self.num_res_blocks):

                # Injections in the Encoder
                if self.vae_loc == "maps":
                    self.vae_projections.append(nn.Linear(self.latent_dim, ch*curr_res*curr_res))
                
                ch = int(mult * self.model_channels)
                self._feature_size += ch
            
            if level != len(self.channel_mult) - 1:
                out_ch = ch

                # Injections in the Encoder
                if self.vae_loc == "maps":
                    self.vae_projections.append(nn.Linear(self.latent_dim, ch*curr_res*curr_res))
        
                ch = out_ch
                ds += 1
                self._feature_size += ch
                curr_res = curr_res // 2
        
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
        
            for i in range(self.num_res_blocks + 1):

                # # Unique projection for each downsample step
                # if self.vae_loc == "maps":
                #     self.vae_projections.append(nn.Linear(self.latent_dim, ch*curr_res*curr_res))

                ch = int(self.model_channels * mult)
                
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    ds -= 1
                    curr_res = curr_res * 2

                self._feature_size += ch


    def vae_injection(self, target, z_sample, index = None):

        #if self.vae_inject == 'add':
        if self.vae_inject in ['add', 'concat']: # temporary to test concat injections at start with the same projection logic

            # Project VAE latent to the target dimension with linear layer
            if self.vae_loc == 'maps':
                injection = self.vae_projections[index](z_sample)
            else:
                injection = self.vae_projection(z_sample)

            # Reshape the injection to match the target
            if self.vae_loc == 'emb':
                # Add the VAE latent to the target
                injection = injection.view(target.shape[0], target.shape[1]) #*2)
            else:
                injection = injection.view(target.shape[0], target.shape[1], target.shape[2], target.shape[3])

            if self.vae_inject == 'add':
                target = target + injection
            elif self.vae_inject == 'concat':
                target = th.concat([target, injection], dim=1)
        
        # Legacy Code for Concat Injection, where we concatenate the VAE latent to the target at every pixel (doesn't work with high dim VAE latents)
        # elif self.vae_inject == 'concat':
        #     # Concatenate the VAE latent to the target pixel-wise
        #     injection = z_sample.unsqueeze(2).unsqueeze(3).repeat(1, 1, target.shape[2], target.shape[3])
        #     target = th.concat([target, injection], dim=1)

        return target



    def forward(self, xt, t, cond=None, prior=None, t2 = None, y=None):

        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(
            t, self.model_channels))
        
        if self.var_timestep:
            emb2 = self.time_embed(timestep_embedding(
            t2, self.model_channels))

            # For now, t difference because its easier to implement
            #emb = emb - emb2

            # Concatenate the VAE latent to the timestep embedding 
            emb = th.cat([emb, emb2], dim=1) 


        # VAE Encoding + Injection
        z_sample, kl_div = vae_encoding(self.vae_encoder, 
                                                latent_dim=self.latent_dim, 
                                                xt=xt, 
                                                emb=emb, 
                                                cond=cond, 
                                                prior=prior)
        
        self.vae_noise = z_sample * self.noise_scale

        
        # Xt Dropout to foster Reliance on VAE injections
        xt = self.xt_dropout_layer(xt)


        # Noise Injection in addition to VAE Injection
        self.perturbation_noise = th.randn_like(xt).to(xt.device) * self.noise_scale

        # Fix sample noise for evaluation
        if self.sample_noise is None:
            self.sample_noise = th.randn_like(xt).to(xt.device) * self.noise_scale 

        # Noise that acts as perturbation to the input - only in training mode
        if self.add_noise: 
            if th.is_grad_enabled():
                #self.counter += 1
                #self.noise_level = min((self.counter / self.max_noise_counter),1)
                #xt = xt + self.perturbation_noise * self.noise_level
                xt = xt + self.perturbation_noise
            else:
                # xt = xt + self.sample_noise[:xt.shape[0]] #* self.noise_level #* 0 # * 0.5
                pass

        # VAE Injection at start 
        if self.vae_loc == 'bold':
            # self.vae_noise = th.randn(xt.shape[0], self.in_channels, self.image_size, self.image_size).to(xt.device) * 0.01
            # self.vae_noise = self.vae_noise.view(xt.shape[0], self.in_channels, self.image_size, self.image_size) # * self.inject_scale
            xt = xt + self.vae_noise.view(xt.shape[0], self.in_channels, self.image_size, self.image_size)

        # VAE Injection at start
        if self.vae_loc == 'start':
            xt = self.vae_injection(xt, z_sample)

        # VAE Injection at timestep embedding
        if self.vae_loc == 'emb':
            emb = self.vae_injection(emb, z_sample)

        # emb += th.randn_like(emb) * 0.01

        # # VAE Injection at start also for maps
        # if self.vae_loc == "maps":
        #     xt = self.vae_injection(xt, z_sample, index = 0)

 
        # xt_noise = th.randn_like(xt) * 0.01
        # xt = th.cat([xt, xt_noise], dim=1)
        
        # xt += th.randn_like(xt) * 0.01
        
        h = xt.type(self.dtype)
        for i, module in enumerate(self.input_blocks):

            # Encoder Injections
            if self.vae_loc == "maps" and i > 0: # For after the first convolution
                h = self.vae_injection(h, z_sample, index = i - 1) 

            h = module(h, emb)
            
            # Encoder Noise injections
            # h += th.randn_like(h) * 0.005

            hs.append(h)

        h = self.middle_block(h, emb)

        # VAE Injection at bottleneck
        if self.vae_loc == 'bottleneck':
            h = self.vae_injection(h, z_sample)

        num_blocks = len(self.output_blocks)
        for j, module in enumerate(self.output_blocks):
            
            # # Decoder Injections
            # if self.vae_loc == "maps":
            #     h = self.vae_injection(h, z_sample, index = j+1) # Same MLPs as downsample, indexed in reverse, +1 because we have one injection at the start

            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        # # Injection added to feature map at every downsample step
        # if self.vae_loc == "maps":
        #     h = self.vae_injection(h, z_sample, index = -1) # Same MLPs as downsample, indexed in reverse
        
        #h = h + th.randn(xt.shape[0], self.model_channels, self.image_size, self.image_size).to(xt.device) * 0.01

        h = h.type(xt.dtype)
        return self.out(h), kl_div
    

            
