"""Auto VAE benchmark: capture latents once, try decoders, check quality, pick best.

Usage:
  python -m mac16_video.bench.vae_bench              # all configs
  python -m mac16_video.bench.vae_bench causal       # only given specs
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
CONFIGS = [
    "preset:auto", "preset:aggressive", "causal",
    "custom:256:64:16:8", "custom:128:32:16:8",
]
MIN_PSNR = 35.0


def fname(spec, ext="npy"):
    return OUT / f"{spec.replace(':', '_')}.{ext}"


def save_mp4(video, path, fps=16):
    import imageio
    import mlx.core as mx
    v = np.array(video[0].astype(mx.float32))
    v = ((np.clip(v, -1, 1) + 1) * 127.5).astype(np.uint8).transpose(1, 2, 3, 0)
    imageio.mimwrite(str(path), list(v), fps=fps, codec="libx264", quality=8)


def worker(spec):
    import mlx.core as mx
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
    from mac16_video.core.memory import peak_gb, reset_peak
    from mac16_video.core.tiling import make_cfg

    z = mx.load(str(LAT))["z"]
    T, H, W = z.shape[-3:]
    if spec.endswith("+lowmem"):
        from mac16_video.stages.decode_vae import enable_lowmem_convs
        enable_lowmem_convs()
        spec = spec[: -len("+lowmem")]
    reset_peak()
    vae = load_vae_decoder(MODEL / "vae.safetensors", None)
    t = time.time()
    if spec == "causal":
        from mac16_video.stages.decode_vae import decode_causal
        video = decode_causal(vae, z)
    else:
        kind, *v = spec.split(":")
        if kind == "preset":
            cfg = (TilingConfig.auto(H * 8, W * 8, (T - 1) * 4 + 1)
                   if v[0] == "auto" else getattr(TilingConfig, v[0])())
        else:
            cfg = make_cfg(*map(int, v))
        video = vae.decode_tiled(z, cfg)
    mx.eval(video)
    secs = time.time() - t
    peak = peak_gb()
    tag = sys.argv[2]
    np.save(fname(tag), np.array(video.astype(mx.float16)))
    save_mp4(video, fname(tag, "mp4"))
    print("RESULT " + json.dumps({"spec": tag, "peak_gb": round(peak, 2),
                                  "secs": round(secs, 1), "frames": int(video.shape[2])}))


def capture():
    OUT.mkdir(exist_ok=True)
    env = {**os.environ, "MAC16_SAVE_LATENTS": str(LAT)}
    cmd = [sys.executable, "-m", "mac16_video.cli", "--model-dir", str(MODEL),
           "--prompt", PROMPT, "--width", "832", "--height", "480",
           "--num-frames", "33", "--steps", "10", "--seed", "42",
           "--tiling", "aggressive", "--output-path", str(OUT / "capture.mp4")]
    subprocess.run(cmd, env=env, check=True)


def psnr(a, b, rng):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 20 * np.log10(rng / np.sqrt(mse))


def best_psnr(ref, b, rng):
    if b.shape == ref.shape:
        return psnr(ref, b, rng)
    n = min(ref.shape[2], b.shape[2])  # different frame counts: try both alignments
    return max(psnr(ref[:, :, :n], b[:, :, :n], rng),
               psnr(ref[:, :, -n:], b[:, :, -n:], rng))


def run(specs):
    OUT.mkdir(exist_ok=True)
    if not LAT.exists():
        print("[bench] capturing latents once (~5 min)...", flush=True)
        capture()
    if "preset:auto" not in specs and not fname("preset:auto").exists():
        specs = ["preset:auto"] + specs
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

    ref = np.load(fname("preset:auto"))
    rng = float(ref.max() - ref.min()) or 1.0
    for r in res:
        f = fname(r["spec"])
        if "error" not in r and f.exists():
            r["psnr"] = round(best_psnr(ref, np.load(f), rng), 1)

    ok = [r for r in res if r.get("psnr") and r["psnr"] >= MIN_PSNR]
    best = min(ok, key=lambda r: r["peak_gb"]) if ok else None
    rows = ["| config | peak GB | decode s | frames | PSNR vs auto |",
            "|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            rows.append(f"| {r['spec']} | failed | - | - | {r['error'][:60]} |")
        else:
            rows.append(f"| {r['spec']} | {r['peak_gb']} | {r['secs']} | "
                        f"{r['frames']} | {r.get('psnr')} |")
    rec = (f"\n**Best:** `{best['spec']}` ({best['peak_gb']} GB, PSNR {best['psnr']} dB)"
           if best else "\n**Best:** none passed quality check")
    table = "\n".join(rows) + "\n" + rec + "\n"
    Path("benchmarks").mkdir(exist_ok=True)
    md = Path("benchmarks/vae_tiling.md")
    md.write_text("# VAE decode benchmark (Wan2.1 1.3B, 832x480, 33f)\n\n" + table)
    print("\n" + table)
    print(f"[bench] saved -> {md}  |  videos -> {OUT}/*.mp4")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        run(sys.argv[1:] or CONFIGS)
