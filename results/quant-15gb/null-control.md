| Arm | Null salt-0 identity | Null (salts 4-7) vs release (0-3): failed only null / only release, McNemar p | Null WER difference [95 % CI] | Release vs full model: WER difference [95 % CI], failed release / full |
|---|---|---|---|---|
| Q | 24/24 | 0 / 0, p 1.000 | +0.29 pp [-0.02 pp, +0.76 pp] | -0.11 pp [-0.36 pp, +0.08 pp], 0 only release / 0 only full |
| QB | 24/24 | 7 / 2, p 0.180 | +3.43 pp [+0.71 pp, +6.78 pp] | -3.31 pp [-8.01 pp, +0.53 pp], 3 only release / 7 only full |

| Variant | Arm | WER mean | Failed | vs pooled release (8 salts): WER difference [95 % CI] | Failed vs pooled rate, Fisher p |
|---|---|---:|---:|---|---|
| release, salts 0-7 | Q | 1.46 % | 0/192 | — | — |
| v1-dense8 | Q | 1.73 % | 0/96 | +0.28 pp [+0.00 pp, +0.72 pp] | 1.000 |
| v2-trim5892 | Q | 1.57 % | 1/96 | +0.11 pp [-0.37 pp, +0.67 pp] | 0.333 |
| v3-stack | Q | 1.37 % | 0/96 | -0.09 pp [-0.46 pp, +0.17 pp] | 1.000 |
| t5900-cover | Q | 1.50 % | 0/96 | +0.05 pp [-0.03 pp, +0.17 pp] | 1.000 |
| t5400-cover | Q | 1.42 % | 0/96 | -0.03 pp [-0.40 pp, +0.22 pp] | 1.000 |
| t6400-cover | Q | 2.85 % | 1/96 | +1.39 pp [-0.44 pp, +4.73 pp] | 0.333 |
| release, salts 0-7 | QB | 4.63 % | 15/192 | — | — |
| v1-dense8 | QB | 5.91 % | 8/96 | +1.28 pp [-1.45 pp, +4.13 pp] | 1.000 |
| v2-trim5892 | QB | 5.41 % | 7/96 | +0.78 pp [-1.81 pp, +3.54 pp] | 1.000 |
| v3-stack | QB | 5.25 % | 11/96 | +0.61 pp [-2.69 pp, +4.07 pp] | 0.383 |
| t5900-cover | QB | 5.85 % | 8/96 | +1.21 pp [-1.45 pp, +3.96 pp] | 1.000 |
| t5400-cover | QB | 7.46 % | 11/96 | +2.83 pp [-1.87 pp, +8.45 pp] | 0.383 |
| t6400-cover | QB | 6.55 % | 9/96 | +1.91 pp [-2.59 pp, +7.56 pp] | 0.656 |
