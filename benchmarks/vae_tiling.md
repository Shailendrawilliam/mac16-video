# VAE decode benchmark (Wan2.1 1.3B, 832x480, 33f)

| config | peak GB | decode s | frames | PSNR vs auto | PSNR vs causal |
|---|---|---|---|---|---|
| causal | 11.17 | 186.1 | 33 | 36.3 | 99.0 |
| causal+bf16 | 5.66 | 32.0 | 33 | 36.1 | 51.0 |
