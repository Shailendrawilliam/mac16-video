"""mac16-video CLI: mlx-video Wan generation with 16 GB-friendly patches.

Usage:
  mac16 "your prompt"                      # 832x480, 33 frames, 10 steps
  mac16 "your prompt" --steps 20 --seed 42 # any mlx-video flag works

Env options:
  MAC16_MODEL=./wan13b-bf16    model dir (default)
  MAC16_VAE=causal|original    VAE decoder (default causal for Wan2.1)
  MAC16_VAE_DTYPE=bf16|fp32    VAE precision (default bf16, ~51 dB vs fp32)
  MAC16_TILING=sp:spov:tf:tfov custom tiling (original VAE only)
  MAC16_SAVE_LATENTS=path      save latents before VAE decode
"""
import atexit
import os
import sys
import time

import mlx.core as mx
import mlx_video.models.wan_2.generate as gen

from mac16_video.core.memory import peak_gb, reset_peak
from mac16_video.stages import encode_text as et


def _defaults():
    return {
        "--model-dir": os.environ.get("MAC16_MODEL", "./wan13b-bf16"),
        "--width": "832",
        "--height": "480",
        "--num-frames": "33",
        "--steps": "10",
        "--output-path": f"mac16_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
    }


def _with_defaults(argv):
    if argv and not argv[0].startswith("-"):
        argv = ["--prompt", argv[0]] + argv[1:]
    for k, v in _defaults().items():
        if k not in argv:
            argv += [k, v]
    return argv


def _stage(name, fn):
    def w(*a, **k):
        print(f"[mac16] prev stage peak {peak_gb():.2f} GB -> start {name}")
        reset_peak()
        return fn(*a, **k)
    return w


def _causal_vae(fn):
    def w(path, config=None, *a, **k):
        vae = fn(path, config, *a, **k)
        if (os.environ.get("MAC16_VAE", "causal") == "causal"
                and getattr(config, "vae_z_dim", 16) == 16):
            from mac16_video.stages.decode_vae import decode_causal, to_bf16
            dtype = os.environ.get("MAC16_VAE_DTYPE", "bf16")
            if dtype == "bf16":
                to_bf16(vae)
            f = lambda z, *aa, **kk: decode_causal(vae, z)
            object.__setattr__(vae, "decode_tiled", f)
            object.__setattr__(vae, "decode", f)
            print(f"[mac16] causal low-memory VAE ({dtype})")
        return vae
    return w


def _capture(fn):
    def w(*a, **k):
        vae = fn(*a, **k)
        path = os.environ.get("MAC16_SAVE_LATENTS")
        if path:
            done = {"v": False}
            for m in ("decode_tiled", "decode"):
                orig = getattr(vae, m, None)
                if orig is None:
                    continue

                def saver(z, *aa, _o=orig, **kk):
                    if not done["v"]:
                        mx.save_safetensors(path, {"z": z})
                        done["v"] = True
                        print(f"[mac16] latents saved -> {path}")
                    return _o(z, *aa, **kk)
                object.__setattr__(vae, m, saver)
        return vae
    return w


def main():
    sys.argv = [sys.argv[0]] + _with_defaults(sys.argv[1:])
    gen.load_t5_encoder = et.load_streaming_t5
    gen.encode_text = et.encode_text
    gen.load_wan_model = _stage("load_wan_model", gen.load_wan_model)
    gen.load_vae_decoder = _stage("load_vae_decoder",
                                  _capture(_causal_vae(gen.load_vae_decoder)))
    spec = os.environ.get("MAC16_TILING")
    if spec:
        from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
        from mac16_video.core.tiling import make_cfg
        cfg = make_cfg(*map(int, spec.split(":")))
        TilingConfig.auto = classmethod(lambda cls, *a, **k: cfg)
        print(f"[mac16] custom tiling {spec}")
    atexit.register(lambda: print(f"[mac16] last stage peak {peak_gb():.2f} GB"))
    gen.main()


if __name__ == "__main__":
    main()
