# CPU Profile Summary (auto-generated)

- Device: `cpu`
- OS: `Windows-11-10.0.26200-SP0`
- CPU: `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD` (physical=unknown, logical=32)
- Software: torch `2.11.0+cpu`, numpy `2.4.2`, python `3.13.1`
- Methodology: warmup=3, repeats=10, timer=`time.perf_counter`, torch_num_threads=32
- Input source: `outputs/pairs_test.npz`
- Final system: threshold=0.3970, model=facenet, dim=512

## Per-batch results (mean)

| batch_size | preprocess_ms | embed_ms | score_ms | end_to_end_ms | throughput_pairs_per_s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.162 ± 0.468 | 79.957 ± 4.572 | 0.210 ± 0.040 | 81.331 ± 4.729 | 12.295 |
| 2 | 2.008 ± 0.358 | 84.607 ± 4.625 | 0.178 ± 0.025 | 86.795 ± 4.438 | 23.043 |
| 4 | 3.255 ± 0.198 | 94.649 ± 2.802 | 0.173 ± 0.015 | 98.078 ± 2.799 | 40.784 |
| 8 | 6.191 ± 0.423 | 123.926 ± 6.700 | 0.178 ± 0.010 | 130.297 ± 6.644 | 61.398 |
| 16 | 12.543 ± 1.198 | 183.999 ± 5.803 | 0.204 ± 0.031 | 196.747 ± 6.487 | 81.323 |
