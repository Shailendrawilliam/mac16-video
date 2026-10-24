# Image-to-video benchmark (Wan2.2 TI2V-5B q8, M5 16 GB)

| date | res | frames | steps | VAE | time | peak footprint |
|---|---|---|---|---|---|---|
| 2026-10-23 | 480x832 | 41 | 10 | aggressive tiling | 10 min | 14.0 GB |
| 2026-10-23 | 480x832 | 41 | 10 | causal bf16 (v0.7b) | 3 min | 17.9 GB |
| 2026-10-23 | 480x832 | 41 | 20 | causal bf16 strips (v0.7c), action prompt | 8 min | 17.8 GB |
| 2026-10-23 | 704x1280 | 81 | 4 (turbo) | causal bf16 | 5 min | 13.3 GB |
| 2026-10-23 | 704x1280 | 121 | 4 (turbo) | causal bf16 | 11 min | 16.2 GB |
