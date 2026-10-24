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

from mac16_video.core.memory import active_gb, cache_gb, peak_gb, reset_peak
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


PRESETS = {
    # measured on M5 16 GB: 704x1280, 41f -> 2.7 min, 13.4 GB. Turbo breaks faces/hands at 480p.
    "turbo720": {"--model-dir": os.environ.get("MAC16_TURBO_MODEL", "./wan22-5b-turbo-q8"),
                 "--width": "704", "--height": "1280", "--num-frames": "41",
                 "--steps": "4", "--guide-scale": "1"},
    # base Wan2.2 5B: 480x832, 41f, 20 steps -> 8.5 min. Fallback when Turbo misbehaves.
    "base480": {"--model-dir": os.environ.get("MAC16_I2V_MODEL", "./wan22-5b-q8"),
                "--width": "480", "--height": "832", "--num-frames": "41", "--steps": "20"},
}


def _apply_preset(argv):
    if "--preset" not in argv:
        return argv
    i = argv.index("--preset")
    name = argv[i + 1] if i + 1 < len(argv) else ""
    argv = argv[:i] + argv[i + 2:]
    if name not in PRESETS:
        raise SystemExit(f"unknown preset '{name}'. Use: {', '.join(PRESETS)}")
    for k, v in PRESETS[name].items():
        if k not in argv:
            argv += [k, v]
    print(f"[mac16] preset {name}")
    return argv


def _with_defaults(argv):
    argv = _apply_preset(argv)
    if "--prompt" not in argv:
        # find the first positional token (not a flag and not a flag's value)
        bool_flags = {"--no-compile", "--no-negative-prompt", "--debug-latents"}
        two_value = {"--lora", "--lora-high", "--lora-low"}
        i = 0
        while i < len(argv):
            t = argv[i]
            if t.startswith("--"):
                i += 1 if t in bool_flags else 3 if t in two_value else 2
                continue
            argv = ["--prompt", t] + argv[:i] + argv[i + 1:]
            break
    for k, v in _defaults().items():
        if k not in argv:
            argv += [k, v]
    # mx.compile gave no speed-up on M5 (143.5 s vs 144.8 s) but pins transformer weights
    if os.environ.get("MAC16_COMPILE") != "1" and "--no-compile" not in argv:
        argv += ["--no-compile"]
    return argv


def _stage(name, fn):
    def w(*a, **k):
        print(f"[mac16] prev stage peak {peak_gb():.2f} GB | now active {active_gb():.2f} GB, cache {cache_gb():.2f} GB -> start {name}")
        reset_peak()
        return fn(*a, **k)
    return w


_MODELS = []


def _track(fn):
    def w(*a, **k):
        m = fn(*a, **k)
        _MODELS.append(m)
        return m
    return w


def _free_models():
    """mlx-video keeps the transformer alive via a loop variable; empty its weights."""
    import gc
    from mlx.utils import tree_map
    for m in _MODELS:
        try:
            if hasattr(m, "_compiled"):
                del m._compiled
            m.update(tree_map(lambda p: mx.zeros((0,), dtype=p.dtype), m.parameters()))
        except Exception as e:
            print(f"[mac16] could not free model weights: {e}")
    _MODELS.clear()
    gc.collect()
    mx.clear_cache()
    print(f"[mac16] transformer weights freed -> active {active_gb():.2f} GB")


def _causal_vae(fn):
    def w(path, config=None, *a, **k):
        _free_models()
        vae = fn(path, config, *a, **k)
        if os.environ.get("MAC16_VAE", "causal") != "causal":
            return vae
        from mac16_video.stages.decode_vae import to_bf16
        dtype = os.environ.get("MAC16_VAE_DTYPE", "bf16")
        zdim = getattr(config, "vae_z_dim", 16)
        if zdim == 16:
            from mac16_video.stages.decode_vae import decode_causal as dec
        elif zdim == 48:
            from mac16_video.stages.decode_vae22 import decode_causal22 as dec
        else:
            return vae
        if dtype == "bf16":
            to_bf16(vae)
        f = lambda z, *aa, **kk: dec(vae, z)
        object.__setattr__(vae, "decode_tiled", f)
        object.__setattr__(vae, "decode", f)
        print(f"[mac16] causal low-memory VAE (Wan{'2.1' if zdim == 16 else '2.2'}, {dtype})")
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
    gen.load_wan_model = _stage("load_wan_model", _track(gen.load_wan_model))
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
