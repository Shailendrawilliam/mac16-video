"""Streaming T5: loads one layer at a time in float32 (exact) + disk cache."""
import hashlib
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_video.models.wan_2.text_encoder import T5SelfAttentionBlock, T5LayerNorm
from mlx_video.models.wan_2 import utils as wan_utils

from mac16_video.core.memory import peak_gb, reset_peak

CACHE = Path(os.environ.get("MAC16_CACHE", Path.home() / ".cache/mac16-video/t5"))


class StreamingT5:
    def __init__(self, path, config):
        self.path, self.cfg = str(path), config


def load_streaming_t5(model_path, config):
    return StreamingT5(model_path, config)


def _encode(enc, ids, mask):
    c = enc.cfg
    w = mx.load(enc.path)
    x = w.pop("token_embedding.weight")[ids].astype(mx.float32)
    mx.eval(x)
    mx.clear_cache()
    for i in range(c.t5_num_layers):
        p = f"blocks.{i}."
        keys = [k for k in w if k.startswith(p)]
        bw = [(k[len(p):], w.pop(k).astype(mx.float32)) for k in keys]
        blk = T5SelfAttentionBlock(c.t5_dim, c.t5_dim_attn, c.t5_dim_ffn,
                                   c.t5_num_heads, c.t5_num_buckets, False)
        blk.load_weights(bw)
        x = blk(x, mask=mask, pos_bias=None)
        mx.eval(x)
        del blk, bw
        mx.clear_cache()
    norm = T5LayerNorm(c.t5_dim)
    norm.load_weights([("weight", w.pop("norm.weight").astype(mx.float32))])
    x = norm(x)
    mx.eval(x)
    return x


def encode_text(encoder, tokenizer, prompt, text_len=512):
    prompt = wan_utils._clean_text(prompt)
    key = hashlib.sha256(
        f"{prompt}|{text_len}|{os.path.getsize(encoder.path)}".encode()
    ).hexdigest()[:16]
    f = CACHE / f"{key}.safetensors"
    if f.exists():
        print(f"[mac16] T5 cache hit ({key})")
        return mx.load(str(f))["emb"]
    tok = tokenizer(prompt, max_length=text_len, padding="max_length",
                    truncation=True, return_tensors="np")
    ids, mask = mx.array(tok["input_ids"]), mx.array(tok["attention_mask"])
    reset_peak()
    t = time.time()
    emb = _encode(encoder, ids, mask)[0, : int(mask.sum().item())]
    mx.eval(emb)
    CACHE.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(f), {"emb": emb})
    print(f"[mac16] T5 streamed in {time.time()-t:.1f}s, peak {peak_gb():.2f} GB")
    return emb
