"""Low-memory Wan2.1 VAE decode.

- enable_lowmem_convs(): CausalConv3d computes one time-step / row-strip at a time.
- decode_causal(): official-style causal decode, one latent frame at a time,
  with every residual-block op evaluated and freed before the next.
- to_bf16(): optional half-precision VAE (about half the memory, tiny quality cost).
All of these except bf16 are mathematically identical to the full decode.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_video.models.wan_2 import vae as wan_vae
from mlx_video.models.wan_2.vae import CACHE_T, ResidualBlock

BUDGET = 128 * 1024 * 1024  # im2col elements per conv call (~512 MB fp32)
_ORIG_CONV3D = wan_vae.CausalConv3d._conv3d
_PATCHED = False


def _conv2d_strips(x, w):
    """Valid stride-1 conv2d split into row strips. x [B,H,W,C], w [O,kh,kw,C]."""
    kh, kw, cin = w.shape[1], w.shape[2], w.shape[3]
    h_out = x.shape[1] - kh + 1
    per_row = x.shape[0] * x.shape[2] * kh * kw * cin
    if h_out * per_row <= BUDGET:
        return mx.conv2d(x, w)
    rows = max(1, BUDGET // per_row)
    parts = []
    for r in range(0, h_out, rows):
        part = mx.conv2d(x[:, r:r + rows + kh - 1], w)
        mx.eval(part)
        parts.append(part)
    return mx.concatenate(parts, axis=1)


def _conv3d_lowmem(self, x):
    b, t, h, w, c_in = x.shape
    kt, kh, kw = self.kernel_size
    st, sh, sw = self.stride
    if (sh, sw) != (1, 1):
        return _ORIG_CONV3D(self, x)
    t_out = (t - kt) // st + 1
    w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(
        self.weight.shape[0], kh, kw, kt * c_in)
    outs = []
    for t_i in range(t_out):
        t0 = t_i * st
        window = x[:, t0:t0 + kt].transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
        out = _conv2d_strips(window, w_2d) + self.bias
        mx.eval(out)
        outs.append(out)
    return mx.stack(outs, axis=1)


def enable_lowmem_convs():
    global _PATCHED
    if not _PATCHED:
        wan_vae.CausalConv3d._conv3d = _conv3d_lowmem
        _PATCHED = True


def to_bf16(vae):
    vae.update(tree_map(lambda p: p.astype(mx.bfloat16), vae.parameters()))
    mx.eval(vae.parameters())
    return vae


def _upsample_conv(conv, x):
    """nn.Conv2d(k=3, padding=1) one frame at a time with row strips."""
    outs = []
    for f in range(x.shape[0]):
        xf = mx.pad(x[f:f + 1], [(0, 0), (1, 1), (1, 1), (0, 0)])
        o = _conv2d_strips(xf, conv.weight) + conv.bias
        mx.eval(o)
        outs.append(o)
    return mx.concatenate(outs, axis=0)


def _cached_conv(conv, x, cache, idx):
    i = idx[0]
    cache_x = x[:, :, -CACHE_T:]
    if cache_x.shape[2] < CACHE_T and isinstance(cache[i], mx.array):
        cache_x = mx.concatenate([cache[i][:, :, -1:], cache_x], axis=2)
    x = conv(x, cache_x=cache[i] if isinstance(cache[i], mx.array) else None)
    cache[i] = cache_x
    idx[0] += 1
    return x


def _resblock(blk, x, cache, idx):
    """ResidualBlock with temporal cache; each op evaluated so temporaries are freed."""
    h = x if blk.shortcut is None else blk.shortcut(x)
    mx.eval(h)
    t = nn.silu(blk.residual[0](x))
    mx.eval(t)
    t = _cached_conv(blk.residual[2], t, cache, idx)
    mx.eval(t)
    t = nn.silu(blk.residual[3](t))
    mx.eval(t)
    t = _cached_conv(blk.residual[6], t, cache, idx)
    out = t + h
    mx.eval(out)
    return out


def _resample_up(layer, x, cache, idx):
    b, c, t, h, w = x.shape
    if layer.mode == "upsample3d":
        i = idx[0]
        if cache[i] is None:
            cache[i] = "Rep"  # first chunk: no temporal upsample (official behaviour)
            idx[0] += 1
        else:
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T:
                prev = (mx.zeros_like(cache_x) if isinstance(cache[i], str)
                        else cache[i][:, :, -1:])
                cache_x = mx.concatenate([prev, cache_x], axis=2)
            if isinstance(cache[i], str):
                x_t = layer.time_conv(x)
            else:
                x_t = layer.time_conv(x, cache_x=cache[i])
            cache[i] = cache_x
            idx[0] += 1
            x_t = x_t.reshape(b, 2, c, t, h, w)
            x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
            mx.eval(x)
            t = t * 2
    x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    x = mx.repeat(x, 2, axis=1)
    x = mx.repeat(x, 2, axis=2)
    x = _upsample_conv(layer.resample[1], x)
    return x.reshape(b, t, h * 2, w * 2, x.shape[-1]).transpose(0, 4, 1, 2, 3)


def decode_causal(vae, z):
    """z: normalized latent [B, 16, T, H, W] -> float32 video [B, 3, 1+4*(T-1), 8H, 8W]."""
    enable_lowmem_convs()
    dec = vae.decoder
    z = z.astype(dec.conv1.weight.dtype)
    mean = vae.mean.reshape(1, -1, 1, 1, 1)
    inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
    x = vae.conv2(z / inv_std + mean)
    mx.eval(x)
    cache = [None] * 64
    outs = []
    for i in range(x.shape[2]):
        idx = [0]
        h = _cached_conv(dec.conv1, x[:, :, i:i + 1], cache, idx)
        mx.eval(h)
        for layer in dec.middle:
            h = _resblock(layer, h, cache, idx) if isinstance(layer, ResidualBlock) else layer(h)
            mx.eval(h)
        for layer in dec.upsamples:
            if isinstance(layer, ResidualBlock):
                h = _resblock(layer, h, cache, idx)
            else:
                h = _resample_up(layer, h, cache, idx)
            mx.eval(h)
        h = nn.silu(dec.head[0](h))
        mx.eval(h)
        h = _cached_conv(dec.head[2], h, cache, idx)
        h = mx.clip(h, -1, 1).astype(mx.float32)
        mx.eval(h, *[c for c in cache if isinstance(c, mx.array)])
        outs.append(h)
        mx.clear_cache()
    return mx.concatenate(outs, axis=2)
