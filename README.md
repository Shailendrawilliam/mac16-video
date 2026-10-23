# mac16-video

**AI video generation tuned for 16 GB Apple Silicon Macs.**

A thin optimization layer on top of [mlx-video](https://github.com/Blaizzy/mlx-video)
that makes Wan2.1 text-to-video run on a 16 GB MacBook without heavy swapping,
with the same output quality.

Tested on: MacBook Air M5, 16 GB unified memory, 512 GB SSD.

## Results (Wan2.1 T2V 1.3B, 832x480, 33 frames, 10 steps)

| Stage | mlx-video | mac16-video | How |
|---|---|---|---|
| T5 text encoder peak | ~22.7 GB | **2.1 GB** | Layer-by-layer streaming in float32 (exact) + prompt cache |
| End-to-end time | 401 s | **277 s** | Less swap |
| VAE decode peak | 19.2 GB | **13.0 GB** | Official-style causal frame-by-frame decode |
| VAE decode time | 174 s | **112 s** | Per-step eval + row-strip convolutions |
| Output frames | 36 | **33** | Causal decode matches official Wan frame count |

Full benchmark: [`benchmarks/vae_tiling.md`](benchmarks/vae_tiling.md)

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install mlx torch git+https://github.com/Blaizzy/mlx-video.git
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
python -m mlx_video.models.wan_2.convert --checkpoint-dir ./Wan2.1-T2V-1.3B --output-dir ./wan13b-bf16
python -m mac16_video.cli --model-dir ./wan13b-bf16 \
  --prompt "A cinematic shot of a futuristic sports car driving through a neon city at night" \
  --width 832 --height 480 --num-frames 33 --steps 10 --output-path out.mp4
```

VAE benchmark (captures latents once, tests decoders, checks PSNR, picks the best):

```bash
python -m mac16_video.bench.vae_bench
```

## Roadmap

- [x] Streaming T5 encoder + prompt cache
- [x] Per-stage memory profiling
- [x] Causal low-memory VAE decode
- [ ] VAE under 8 GB (bf16 / causal + spatial tiles)
- [ ] Causal VAE wired into the main CLI
- [ ] Wan2.2 TI2V-5B (text + image to video)
- [ ] Layer streaming for 14B models
- [ ] Simple UI + job queue for overnight batches

## Credits

Built on [mlx-video](https://github.com/Blaizzy/mlx-video) (MIT) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache 2.0).

## License

Apache-2.0
