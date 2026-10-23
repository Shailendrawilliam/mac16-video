# mac16-video

**AI video generation tuned for 16 GB Apple Silicon Macs.**

A thin optimization layer on top of [mlx-video](https://github.com/Blaizzy/mlx-video)
that runs Wan2.1 text-to-video on a 16 GB MacBook without heavy swapping.

Tested on: MacBook Air M5, 16 GB unified memory, 512 GB SSD.

## Results (Wan2.1 T2V 1.3B, 832x480, 33 frames, 10 steps)

| | mlx-video | mac16-video | How |
|---|---|---|---|
| **End-to-end peak memory** | 25.7 GB | **13.1 GB** | All of the below |
| **End-to-end time** | 401 s | **220 s** | Less swap, faster VAE |
| T5 text encoder peak | ~22.7 GB | 2.1 GB | Layer-by-layer streaming (float32, exact) + prompt cache |
| VAE decode peak | 19.2 GB | 5.7 GB | Official-style causal frame-by-frame decode, bf16 |
| VAE decode time | 174 s | 32 s | Per-op eval, row-strip convs, bf16 |
| Output frames | 36 | 33 | Causal decode matches official Wan frame count |

bf16 VAE vs float32 causal VAE: **51 dB PSNR** (visually identical).
Use `MAC16_VAE_DTYPE=fp32` for exact float32 (11.2 GB VAE peak).
Full benchmark: [`benchmarks/vae_tiling.md`](benchmarks/vae_tiling.md)

## Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "mac16-video[convert] @ git+https://github.com/Shailendrawilliam/mac16-video.git"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
python -m mlx_video.models.wan_2.convert --checkpoint-dir ./Wan2.1-T2V-1.3B --output-dir ./wan13b-bf16
```

## Usage

```bash
mac16 "A cinematic shot of a futuristic sports car driving through a neon city at night"
mac16 "ocean waves at sunset, cinematic" --steps 20 --seed 42 --output-path waves.mp4
```

Defaults: 832x480, 33 frames, 10 steps, model `./wan13b-bf16` (override with `MAC16_MODEL`).
Any mlx-video flag works (`--steps`, `--seed`, `--negative-prompt`, `--trim-first-frames`, ...).

VAE benchmark (captures latents once, tests decoders, checks PSNR):

```bash
mac16-bench                       # default set
mac16-bench causal causal+bf16    # specific decoders
```

## Image-to-video (Wan2.2 TI2V-5B, 8-bit) — measured on M5 16 GB

| | before (mlx-video) | mac16-video |
|---|---|---|
| Wan2.2 VAE decode peak (480x832, 41f) | 11.5 GB (causal fp32 path) / 8.3 GB tiled | **5.4 GB** |
| Wan2.2 VAE decode time | 336 s (tiled) | **33 s** |
| Transformer memory left during VAE | 5.4 GB | **0.07 GB** |
| End-to-end 480x832, 41f, 10 steps | 10.4 min, 14.0 GB | **3.2 min, 13.2 GB** |
| End-to-end 704x1280, 41f, 20 steps | - | 21.7 min, 13.5 GB |

How: causal frame-by-frame Wan2.2 VAE decode, bf16-safe fused RMS norm, cache stored as
copies (not views), transformer weights freed before the VAE, `mx.compile` off by default
(no speed difference on M5, but it pinned the weights).

```bash
MAC16_MODEL=./wan22-5b-q8 mac16 "the man looks into the camera and speaks, raises his hand" \
  --image scene.jpg --width 480 --height 832 --num-frames 41 --steps 20
```

## Roadmap

- [x] Streaming T5 encoder + prompt cache
- [x] Per-stage memory profiling
- [x] Causal low-memory VAE decode (bf16 default)
- [x] `mac16` command with sensible defaults
- [x] Wan2.2 TI2V-5B (text + image to video)
- [ ] Layer streaming for 14B models
- [ ] Simple UI + job queue for overnight batches

## Credits

Built on [mlx-video](https://github.com/Blaizzy/mlx-video) (MIT) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache 2.0).

## License

Apache-2.0
