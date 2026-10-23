# VAE decode benchmark (Wan2.1 1.3B, 832x480, 33f)

| config | peak GB | decode s | frames | PSNR vs auto |
|---|---|---|---|---|
| causal | 13.02 | 111.9 | 33 | 36.3 |
| preset:auto+lowmem | 19.22 | 92.1 | 36 | 78.5 |

**Best:** `causal` (13.02 GB, PSNR 36.3 dB)
