"""Auto VAE benchmark: capture latents once, try decoders, check quality, pick best.

Specs: preset:<name>, custom:sp:spov:tf:tfov, causal. Modifiers: +lowmem, +bf16.
Usage:
  python -m mac16_video.bench.vae_bench                    # default set
  python -m mac16_video.bench.vae_bench causal causal+bf16 # only these
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
MODEL = ROOT / "wan13b-bf16"
OUT = ROOT / "bench_out"
LAT = OUT / "latents.safetensors"
PROMPT = "A cinematic shot of a futuristic sports car driving through a neon city at night"
CONFIGS = ["preset:auto", "preset:aggressive", "causal", "causal+bf16"]


def fname(spec, ext="npy"):
    return OUT / f"{spec.replace(':', '_').replace('+', '-')}.{ext}"


def save_mp4(video, path, fps=16):
    import imageio
    import mlx.core as mx
    v = np.array(video[0].astype(mx.float32))
    v = ((np.clip(v, -1, 1) + 1) * 127.5).astype(np.uint8).transpose(1, 2, 3, 0)
    imageio.mimwrite(str(path), list(v), fps=fps, codec="libx264", quality=8)


def worker(full):
    import mlx.core as mx
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
    from mac16_video.core.memory import peak_gb, reset_peak
    from mac16_video.core.tiling import make_cfg
    from mac16_video.stages.decode_vae import decode_causal, enable_lowmem_convs, to_bf16

    spec, *mods = full.split("+")
    z = mx.load(str(LAT))["z"]
    T, H, W = z.shape[-3:]
    if "lowmem" in mods:
        enable_lowmem_convs()
    reset_peak()
    vae = load_vae_decoder(MODEL / "vae.safetensors", None)
    if "bf16" in mods:
        to_bf16(vae)
    t = time.time()
    if spec == "causal":
        video = decode_causal(vae, z)
    else:
        kind, *v = spec.split(":")
        if kind == "preset":
            cfg = (TilingConfig.auto(H * 8, W * 8, (T - 1) * 4 + 1)
                   if v[0] == "auto" else getattr(TilingConfig, v[0])())
        else:
            cfg = make_cfg(*map(int, v))
        video = vae.decode_tiled(z.astype(vae.decoder.conv1.weight.dtype), cfg)
    mx.eval(video)
    secs = time.time() - t
    peak = peak_gb()
    video = video.astype(mx.float32)
    np.save(fname(full), np.array(video.astype(mx.float16)))
    save_mp4(video, fname(full, "mp4"))
    print("RESULT " + json.dumps({"spec": full, "peak_gb": round(peak, 2),
                                  "secs": round(secs, 1), "frames": int(video.shape[2])}))


def capture():
    OUT.mkdir(exist_ok=True)
    env = {**os.environ, "MAC16_SAVE_LATENTS": str(LAT), "MAC16_VAE": "original"}
    cmd = [sys.executable, "-m", "mac16_video.cli", "--model-dir", str(MODEL),
           "--prompt", PROMPT, "--width", "832", "--height", "480",
           "--num-frames", "33", "--steps", "10", "--seed", "42",
           "--tiling", "aggressive", "--output-path", str(OUT / "capture.mp4")]
    subprocess.run(cmd, env=env, check=True)


def psnr(a, b, rng):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 20 * np.log10(rng / np.sqrt(mse))


def best_psnr(ref, b):
    rng = float(ref.max() - ref.min()) or 1.0
    if b.shape == ref.shape:
        return psnr(ref, b, rng)
    n = min(ref.shape[2], b.shape[2])
    return max(psnr(ref[:, :, :n], b[:, :, :n], rng),
               psnr(ref[:, :, -n:], b[:, :, -n:], rng))


def run(specs):
    OUT.mkdir(exist_ok=True)
    if not LAT.exists():
        print("[bench] capturing latents once (~5 min)...", flush=True)
        capture()
    res = []
    for s in specs:
        print(f"[bench] {s} ...", flush=True)
        try:
            p = subprocess.run([sys.executable, "-m", "mac16_video.bench.vae_bench",
                                "--worker", s], capture_output=True, text=True,
                               timeout=1800)
            out = p.stdout + "\n" + p.stderr
        except subprocess.TimeoutExpired:
            out = "timeout"
        line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            err = (out.strip().splitlines() or ["unknown error"])[-1]
            print(f"   failed: {err}")
            res.append({"spec": s, "error": err})
            continue
        r = json.loads(line[7:])
        res.append(r)
        print(f"   peak {r['peak_gb']} GB, {r['secs']} s, {r['frames']} frames", flush=True)

    refs = {name: np.load(fname(name)) for name in ("preset:auto", "causal")
            if fname(name).exists()}
    for r in res:
        f = fname(r["spec"])
        if "error" not in r and f.exists():
            b = np.load(f)
            for name, ref in refs.items():
                r[f"psnr_{name}"] = round(best_psnr(ref, b), 1)

    rows = ["| config | peak GB | decode s | frames | PSNR vs auto | PSNR vs causal |",
            "|---|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            rows.append(f"| {r['spec']} | failed | - | - | {r['error'][:50]} | - |")
        else:
            rows.append(f"| {r['spec']} | {r['peak_gb']} | {r['secs']} | {r['frames']} | "
                        f"{r.get('psnr_preset:auto')} | {r.get('psnr_causal')} |")
    table = "\n".join(rows) + "\n"
    Path("benchmarks").mkdir(exist_ok=True)
    md = Path("benchmarks/vae_tiling.md")
    md.write_text("# VAE decode benchmark (Wan2.1 1.3B, 832x480, 33f)\n\n" + table)
    print("\n" + table)
    print(f"[bench] saved -> {md}  |  videos -> {OUT}/*.mp4")


def cli():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        run(sys.argv[1:] or CONFIGS)


if __name__ == "__main__":
    cli()
