#!/usr/bin/env python3
"""exllamav3's own quantize_exl3 on card 35's six experts, at every EXL3 bitrate, on the Spark's GPU.

  exllamav3_job.py CHECKPOINT HESSIANS_DIR OUT.json

W is each projection's FP8 original from CHECKPOINT, checked against the sha256 in HESSIANS_DIR/LAYER-EXPERT.npz
(measure.py hessians), H its Hessians from there. Each projection is quantized as exllamav3's converter does by default
(mul1 codebook, automatic output scales, seed 0), from a fresh capture so no sign flips are shared, and its error is
trace((W - Q) H (W - Q)^T) in float64 on the undamped H, the Q exllamav3 returns after refit_scales. OUT.json maps
"LAYER/EXPERT" to {bits: {projection: error}}, the input of measure.py parity.
"""
import importlib.util
import json
import sys
from pathlib import Path
import types

import numpy as np
import torch

import measure
from measure import ce, ex


def import_quantize():
    """exllamav3's quantize_exl3 without running the package __init__s above it, which load the model, tokenizer and
    generator stack (tokenizers, llguidance, ...) the quantizer does not use."""
    root = Path(importlib.util.find_spec('exllamav3').submodule_search_locations[0])
    for package in ('exllamav3', 'exllamav3.modules', 'exllamav3.modules.quant', 'exllamav3.modules.quant.exl3_lib'):
        module = sys.modules[package] = types.ModuleType(package)
        module.__path__ = [str(root.joinpath(*package.split('.')[1:]))]
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    return quantize_exl3


def full(triangle):
    n = int((np.sqrt(8 * len(triangle) + 1) - 1) / 2)
    h = np.zeros((n, n))
    h[np.triu_indices(n)] = triangle
    return h + np.triu(h, 1).T


def quantize(quantize_exl3, w, h, bits):
    """exllamav3's Q of W [out, in] at BITS against H, in W's layout: a fresh capture each time, so no sign flips are
    shared, and Q in the original basis with its scales refit (quantize_exl3 works on [in, out])."""
    capture = {'H': torch.tensor(h, dtype=torch.float32, device='cuda'), 'count': 1, 'finalized': False}
    k = float(bits)
    # whole bitrates index kernels and shift by K, so they must be ints; half-integers take the fractional trellis
    args = {'seed': 0, 'K': int(k) if k.is_integer() else k, 'devices': [0], 'apply_out_scales': None, 'mul1': True}
    q, _, _ = quantize_exl3(torch.tensor(w.T, device='cuda'), capture, args, return_weight_q=True)
    return q.T.double().cpu().numpy()


def main(checkpoint, hessians, out):
    quantize_exl3, source, result = import_quantize(), ce.Tensors(checkpoint, np), {}
    for layer, expert in measure.PROBES:
        stored = np.load(hessians / f'{layer}-{expert}.npz')
        errors = result[f'{layer}/{expert}'] = {bits: {} for bits in ex.BITS}
        for p in ex.PROJECTIONS:
            w = ce.original(source, ex.name(layer, expert, p), np)
            if measure.weight_sha256(w) != str(stored[f'{p}_sha256']):
                sys.exit(f'{layer}/{expert} {p}: W differs from the one the Hessians were made with')
            h = full(stored['h_down' if p == 'down_proj' else 'h_in'].astype(np.float64))
            for bits in ex.BITS:
                errors[bits][p] = ex.hessian_error(w, quantize(quantize_exl3, w, h, bits), h)
                print(f'{layer}/{expert} {p} {bits}: {errors[bits][p]:.6g}', flush=True)
    out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
