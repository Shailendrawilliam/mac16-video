"""Causal low-memory decode for the Wan2.2 VAE (z_dim=48, channels-last).

Mirrors the official Wan2.2 chunked decode: one latent frame per step with a
temporal feature cache (T latents -> 1 + 4*(T-1) frames). Every conv time-step is
evaluated before the next, so peak memory is about one chunk instead of the video.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_video.models.wan_2 import vae22 as V
from mlx_video.models.wan_2.vae22 import CACHE_T, ResidualBlock, Resample, _unpatchify
from mac16_video.stages.decode_vae import _conv2d_strips

import os
from mac16_video.core.memory import active_gb, peak_gb, reset_peak

_PROF = os.environ.get("MAC16_PROFILE") == "1"


def _p(tag):
    if _PROF:
        print(f"[prof] {tag:24s} active {active_gb():5.2f} GB   peak-since-last {peak_gb():5.2f} GB", flush=True)
        reset_peak()


_ORIG_CALL = V.CausalConv3d.__call__
_PATCHED = False


def _conv_lowmem(self, x, cache_x=None):
    B, T, H, W, C = x.shape
    kd, kh, kw = self.kernel_size
    if kd == 1 and kh == 1 and kw == 1:
        return _ORIG_CALL(self, x, cache_x)
    pad_needed = self._causal_pad_t
    if cache_x is not None and pad_needed > 0:
        x = mx.concatenate([cache_x, x], axis=1)
        pad_needed -= cache_x.shape[1]
    if pad_needed > 0:
        x = mx.concatenate([mx.zeros((B, pad_needed, H, W, C), dtype=x.dtype), x], axis=1)
    if self._pad_h > 0 or self._pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (self._pad_h, self._pad_h),
                       (self._pad_w, self._pad_w), (0, 0)])
    mx.eval(x)
    t_out = (x.shape[1] - kd) // self.stride[0] + 1
    outs = []
    for t in range(t_out):
        t0 = t * self.stride[0]
        acc = None
        for d in range(kd):
            if (self.stride[1], self.stride[2]) == (1, 1):
                o = _conv2d_strips(x[:, t0 + d], self.weight[:, d])
            else:
                o = mx.conv_general(x[:, t0 + d], self.weight[:, d],
                                    stride=(self.stride[1], self.stride[2]))
            acc = o if acc is None else acc + o
            mx.eval(acc)
        acc = acc + self.bias
        mx.eval(acc)
        outs.append(acc)
    return mx.stack(outs, axis=1)


def _rms22(self, x):
    return mx.fast.rms_norm(x, self.gamma.astype(x.dtype), 1e-12)


def enable_lowmem22():
    global _PATCHED
    if not _PATCHED:
        V.CausalConv3d.__call__ = _conv_lowmem
        V.RMS_norm.__call__ = _rms22
        _PATCHED = True


def _cached(conv, x, cache, idx):
    j = idx[0]
    cx = x[:, -CACHE_T:]
    if cx.shape[1] < CACHE_T and isinstance(cache[j], mx.array):
        cx = mx.concatenate([cache[j][:, -1:], cx], axis=1)
    x = conv(x, cache_x=cache[j] if isinstance(cache[j], mx.array) else None)
    cache[j] = cx * 1
    idx[0] += 1
    return x


def _resblock22(blk, x, cache, idx):
    h = blk.shortcut(x) if blk.shortcut is not None else x
    mx.eval(h)
    r = blk.residual
    t = nn.silu(r.layer_0(x))
    mx.eval(t)
    t = _cached(r.layer_2, t, cache, idx)
    mx.eval(t)
    t = nn.silu(r.layer_3(t))
    mx.eval(t)
    t = _cached(r.layer_6, t, cache, idx)
    out = t + h
    mx.eval(out)
    return out


def _resample_up(layer, x, cache, idx):
    B, T, H, W, C = x.shape
    if layer.mode == "upsample3d":
        i = idx[0]
        if cache[i] is None:
            cache[i] = "Rep"  # first chunk: no temporal upsample (official behaviour)
            idx[0] += 1
        else:
            cx = x[:, -CACHE_T:]
            if cx.shape[1] < CACHE_T:
                prev = mx.zeros_like(cx) if isinstance(cache[i], str) else cache[i][:, -1:]
                cx = mx.concatenate([prev, cx], axis=1)
            tc = layer.time_conv(x) if isinstance(cache[i], str) else layer.time_conv(x, cache_x=cache[i])
            cache[i] = cx * 1
            idx[0] += 1
            tc = tc.reshape(B, T, H, W, 2, C)
            x = mx.stack([tc[:, :, :, :, 0, :], tc[:, :, :, :, 1, :]], axis=2).reshape(B, T * 2, H, W, C)
            mx.eval(x)
            T = T * 2
    frames = []
    for f in range(T):
        up = mx.pad(layer._upsample2x(x[:, f]), [(0, 0), (1, 1), (1, 1), (0, 0)])
        xf = _conv2d_strips(up, layer.resample_weight) + layer.resample_bias
        mx.eval(xf)
        frames.append(xf)
    return mx.stack(frames, axis=1)


def _up_block(blk, x, cache, idx, first_chunk):
    h = x
    for m in blk.upsamples:
        h = _resample_up(m, h, cache, idx) if isinstance(m, Resample) else _resblock22(m, h, cache, idx)
        mx.eval(h)
    if blk.avg_shortcut is not None:
        sc = blk.avg_shortcut(x, first_chunk)
        mx.eval(sc)
        return h + sc
    return h


def decode_causal22(vae, z):
    """z: denormalized latents [B, T, H, W, 48] -> float32 video [B, 1+4*(T-1), 16H, 16W, 3]."""
    enable_lowmem22()
    dec = vae.decoder
    _p('decode start')
    x = vae.conv2(z.astype(dec.conv1.weight.dtype))
    mx.eval(x)
    _p('conv2')
    cache = [None] * 128
    outs = []
    for i in range(x.shape[1]):
        idx = [0]
        h = _cached(dec.conv1, x[:, i:i + 1], cache, idx)
        mx.eval(h)
        if i < 2: _p(f'c{i} conv1')
        for layer in dec.middle:
            h = _resblock22(layer, h, cache, idx) if isinstance(layer, ResidualBlock) else layer(h)
            mx.eval(h)
        if i < 2: _p(f'c{i} middle')
        for k, blk in enumerate(dec.upsamples):
            h = _up_block(blk, h, cache, idx, first_chunk=(i == 0))
            mx.eval(h)
            if i < 2: _p(f'c{i} up{k} {tuple(h.shape)}')
        h = _cached(dec.head.layer_2, nn.silu(dec.head.layer_0(h)), cache, idx)
        h = mx.clip(_unpatchify(h, patch_size=2), -1.0, 1.0).astype(mx.float32)
        mx.eval(h, *[c for c in cache if isinstance(c, mx.array)])
        outs.append(h)
        mx.clear_cache()
        _p(f'c{i} done')
    return mx.concatenate(outs, axis=1)
