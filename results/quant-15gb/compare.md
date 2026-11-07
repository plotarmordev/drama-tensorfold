### Reference voice, guidance γ 1.5 batched (arm Q)

| Measure | v0-release | release | v2-trim5892 | t5900-cover | t5400-cover | t6400-cover |
|---|---:|---:|---:|---:|---:|---:|
| Weights + codec on disk (GiB) | 20.76 | 20.76 | 18.33 | 18.34 | 17.52 | 19.17 |
| Peak device memory, /health (GiB) | 19.35 | 19.35 | 16.92 | 16.93 | 16.11 | 17.75 |
| Host MemAvailable drop, max (GiB) | 21.57 | 21.27 | 19.03 | 18.79 | 36.04 | 27.61 |
| Engine load, median (s) | 25.6 | 26.1 | 18.8 | 20.1 | 22.9 | 24.4 |
| RTF warm p50 / p95, salt 0 | 0.818 / 1.433 | 0.825 / 1.462 | 0.812 / 1.322 | 0.828 / 1.342 | 0.824 / 1.498 | 0.883 / 1.566 |
| RTF per-salt first warm pass, low–high | 0.826–0.850 | 0.824–0.838 | 0.811–0.820 | 0.820–0.825 | 0.827–0.905 | 0.858–0.896 |
| Whole reply warm p50 (s) | 8.21 | 8.38 | 8.26 | 8.45 | 7.96 | 10.53 |
| First audio streamed p50 (s) | 1.941 | 1.955 | 1.875 | 1.824 | 1.785 | 1.958 |
| Decode codec frames/s, streamed | 31.9 | 32.0 | 32.8 | 32.0 | 31.9 | 28.7 |
| Warm requests succeeded | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed |
| WER mean [95 % CI] | 1.31 % [0.04, 3.32] | 1.31 % [0.04, 3.32] | 1.57 % [0.06, 4.07] | 1.50 % [0.06, 3.59] | 1.42 % [0.12, 3.45] | 2.85 % [0.17, 6.73] |
| WER pooled | 0.56 % | 0.56 % | 0.63 % | 0.71 % | 0.71 % | 1.94 % |
| Failed takes (capped, collapsed) | 0/96 (0, 0) | 0/96 (0, 0) | 1/96 (0, 1) | 0/96 (0, 0) | 0/96 (0, 0) | 1/96 (0, 1) |
| Failed per salt 0/1/2/3 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 1 |
| Engine finish | stop 96 | stop 96 | stop 96 | stop 96 | stop 96 | stop 96 |
| Speaker cosine to full model | 0.900 | 0.900 | 0.900 | 0.895 | 0.895 | 0.898 |
| UTMOS mean (p10) | 4.506 (4.441) | 4.506 (4.441) | 4.502 (4.441) | 4.505 (4.452) | 4.511 (4.468) | 4.507 (4.454) |
| Audio length / full model, median | 0.969 | 0.969 | 0.991 | 1.000 | 0.994 | 1.029 |
| Pitch vs full, median abs semitones | 0.619 | 0.619 | 0.821 | 0.700 | 0.698 | 0.766 |

| Paired with the release (same items, salts) | Pairs | Byte-identical takes | Failed only in variant | Failed only in release | Exact McNemar p | WER difference, mean [95 % CI over items] |
|---|---:|---:|---:|---:|---:|---|
| release | 96 | 96 | 0 | 0 | 1.000 | +0.00 pp [+0.00, +0.00] |
| v2-trim5892 | 96 | 0 | 1 | 0 | 1.000 | +0.26 pp [-0.11, +0.78] |
| t5900-cover | 96 | 0 | 0 | 0 | 1.000 | +0.19 pp [-0.02, +0.49] |
| t5400-cover | 96 | 0 | 0 | 0 | 1.000 | +0.11 pp [-0.09, +0.36] |
| t6400-cover | 96 | 0 | 1 | 0 | 1.000 | +1.54 pp [-0.13, +4.75] |

### Built-in voice (arm QB)

| Measure | v0-release | release | v2-trim5892 | t5900-cover | t5400-cover | t6400-cover |
|---|---:|---:|---:|---:|---:|---:|
| Weights + codec on disk (GiB) | 20.76 | 20.76 | 18.33 | 18.34 | 17.52 | 19.17 |
| Peak device memory, /health (GiB) | 19.35 | 19.35 | 16.92 | 16.93 | 16.11 | 17.75 |
| Host MemAvailable drop, max (GiB) | 21.18 | 21.23 | 18.65 | 19.14 | 25.24 | 22.40 |
| Engine load, median (s) | 25.0 | 26.5 | 20.9 | 20.1 | 22.1 | 27.0 |
| RTF warm p50 / p95, salt 0 | 0.606 / 0.642 | 0.618 / 0.656 | 0.605 / 0.638 | 0.615 / 0.660 | 0.681 / 0.731 | 0.670 / 0.709 |
| RTF per-salt first warm pass, low–high | 0.606–0.609 | 0.611–0.621 | 0.598–0.603 | 0.614–0.616 | 0.677–0.700 | 0.664–0.678 |
| Whole reply warm p50 (s) | 5.26 | 5.37 | 4.99 | 5.03 | 5.51 | 6.03 |
| First audio streamed p50 (s) | 0.112 | 0.113 | 0.109 | 0.106 | 0.113 | 0.119 |
| Decode codec frames/s, streamed | 34.8 | 34.1 | 35.1 | 34.3 | 31.0 | 31.9 |
| Warm requests succeeded | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed | 72/72 + 72/72 streamed |
| WER mean [95 % CI] | 2.92 % [0.24, 6.57] | 2.92 % [0.24, 6.57] | 5.41 % [2.13, 9.37] | 5.85 % [1.65, 11.22] | 7.46 % [2.57, 13.56] | 6.55 % [2.25, 11.83] |
| WER pooled | 1.01 % | 1.01 % | 3.06 % | 4.48 % | 9.29 % | 8.51 % |
| Failed takes (capped, collapsed) | 5/96 (0, 5) | 5/96 (0, 5) | 7/96 (0, 7) | 8/96 (0, 8) | 11/96 (0, 11) | 9/96 (0, 9) |
| Failed per salt 0/1/2/3 | 0 / 2 / 2 / 1 | 0 / 2 / 2 / 1 | 2 / 1 / 2 / 2 | 2 / 0 / 2 / 4 | 5 / 3 / 2 / 1 | 1 / 3 / 2 / 3 |
| Engine finish | silence 2, stop 94 | silence 2, stop 94 | silence 5, stop 91 | silence 6, stop 90 | silence 8, stop 88 | silence 5, stop 91 |
| Speaker cosine to full model | 0.308 | 0.308 | 0.315 | 0.312 | 0.295 | 0.300 |
| UTMOS mean (p10) | 4.486 (4.413) | 4.486 (4.413) | 4.469 (4.385) | 4.438 (4.290) | 4.477 (4.350) | 4.497 (4.387) |
| Audio length / full model, median | 0.971 | 0.971 | 0.986 | 0.982 | 0.954 | 1.002 |
| Pitch vs full, median abs semitones | 3.492 | 3.492 | 3.699 | 3.933 | 4.869 | 3.641 |

| Paired with the release (same items, salts) | Pairs | Byte-identical takes | Failed only in variant | Failed only in release | Exact McNemar p | WER difference, mean [95 % CI over items] |
|---|---:|---:|---:|---:|---:|---|
| release | 96 | 96 | 0 | 0 | 1.000 | +0.00 pp [+0.00, +0.00] |
| v2-trim5892 | 96 | 0 | 6 | 4 | 0.754 | +2.50 pp [+0.41, +4.99] |
| t5900-cover | 96 | 0 | 7 | 4 | 0.549 | +2.93 pp [+0.43, +5.81] |
| t5400-cover | 96 | 0 | 9 | 3 | 0.146 | +4.54 pp [-0.43, +10.82] |
| t6400-cover | 96 | 0 | 9 | 5 | 0.424 | +3.63 pp [-0.74, +8.98] |

