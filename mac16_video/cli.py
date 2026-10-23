import atexit
import os

import mlx.core as mx
import mlx_video.models.wan_2.generate as gen

from mac16_video.core.memory import peak_gb, reset_peak
from mac16_video.stages import encode_text as et


def _stage(name, fn):
    def w(*a, **k):
        print(f"[mac16] prev stage peak {peak_gb():.2f} GB -> start {name}")
        reset_peak()
        return fn(*a, **k)
    return w


def _capture(fn):
    """If MAC16_SAVE_LATENTS is set, save latents passed to the VAE (once)."""
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
    gen.load_t5_encoder = et.load_streaming_t5
    gen.encode_text = et.encode_text
    if hasattr(gen, "load_wan_model"):
        gen.load_wan_model = _stage("load_wan_model", gen.load_wan_model)
    if hasattr(gen, "load_vae_decoder"):
        gen.load_vae_decoder = _stage("load_vae_decoder", _capture(gen.load_vae_decoder))
    spec = os.environ.get("MAC16_TILING")  # e.g. "256:64:16:8", used with --tiling auto
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
