"""Wan2.2 VAE decode bench on captured latents: aggressive tiling vs causal (fp32/bf16)."""
import json, subprocess, sys, time, types
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
MODEL = ROOT / "wan22-5b-q8"
OUT = ROOT / "bench_out"
LAT = OUT / "latents22.safetensors"
SPECS = ["causal", "causal+bf16", "aggressive"]


def worker(spec):
    import mlx.core as mx
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig
    from mac16_video.core.memory import peak_gb, reset_peak
    from mac16_video.stages.decode_vae import to_bf16
    from mac16_video.stages.decode_vae22 import decode_causal22
    z = mx.load(str(LAT))["z"]
    reset_peak()
    vae = load_vae_decoder(MODEL / "vae.safetensors", types.SimpleNamespace(vae_z_dim=48))
    from mac16_video.stages.decode_vae22 import _p
    _p('vae loaded (fp32)')
    if spec.endswith("+bf16"):
        to_bf16(vae)
        _p('vae cast to bf16')
    t = time.time()
    video = (vae.decode_tiled(z, TilingConfig.aggressive()) if spec == "aggressive"
             else decode_causal22(vae, z))
    mx.eval(video)
    secs, peak = time.time() - t, peak_gb()
    np.save(OUT / f"vae22_{spec.replace('+', '-')}.npy", np.array(video.astype(mx.float16)))
    print("RESULT " + json.dumps({"spec": spec, "peak_gb": round(peak, 2),
                                  "secs": round(secs, 1), "frames": int(video.shape[1])}))


def psnr(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    n = min(a.shape[1], b.shape[1]); a, b = a[:, :n], b[:, :n]
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else round(20 * np.log10(2.0 / np.sqrt(mse)), 1)


def run(specs):
    res = []
    for s in specs:
        print(f"[vae22] {s} ...", flush=True)
        p = subprocess.run([sys.executable, "-m", "mac16_video.bench.vae22_bench", "--worker", s],
                           capture_output=True, text=True, timeout=3600)
        out = p.stdout + "\n" + p.stderr
        line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            err = (out.strip().splitlines() or ["unknown error"])[-1]
            print("   failed:", err); res.append({"spec": s, "error": err}); continue
        r = json.loads(line[7:]); res.append(r)
        print(f"   peak {r['peak_gb']} GB, {r['secs']} s, {r['frames']} frames", flush=True)
    ref_f = OUT / "vae22_causal.npy"
    ref = np.load(ref_f) if ref_f.exists() else None
    rows = ["| decode | peak GB | time s | frames | PSNR vs causal fp32 |", "|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            rows.append(f"| {r['spec']} | failed | - | - | {r['error'][:50]} |"); continue
        f = OUT / f"vae22_{r['spec'].replace('+', '-')}.npy"
        ps = psnr(ref, np.load(f)) if ref is not None and f.exists() else None
        rows.append(f"| {r['spec']} | {r['peak_gb']} | {r['secs']} | {r['frames']} | {ps} |")
    table = "\n".join(rows)
    Path("benchmarks/vae22.md").write_text("# Wan2.2 VAE decode (TI2V-5B, 480x832, 41f)\n\n" + table + "\n")
    print("\n" + table + "\n[vae22] saved -> benchmarks/vae22.md")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        run(sys.argv[1:] or SPECS)
