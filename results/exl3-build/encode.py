#!/usr/bin/env python3
"""Every kept routed expert of t5900-cover at EXL3 3.0 bits by exllamav3 bd1f519's quantize_exl3, on a GB10.

  encode.py CHECKPOINT ROWS OUT_DIR

W is each projection's FP8 original from CHECKPOINT, H its expert's Hessians from the NVFP4-mix rows in ROWS
(exl3-calibration-rows-v3, recipe/exl3_experts.py's recipe: an expert with no rows takes all of its layer's), built
on the GPU. Each projection is quantized as exllamav3_job.py does for card 39. A layer is written as
OUT_DIR/layer-LL.safetensors (exllamav3's trellis, suh, svh and mul1 per projection, under the projection's name) and
then OUT_DIR/layer-LL.json (rows, seconds and each projection's error trace((W - Q) H (W - Q)^T) per expert); a layer
with its record is done and is skipped, so a run that stops resumes at the first unfinished layer.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'recipe'))
import calibrate_experts as ce  # noqa: E402
import exl3_experts as ex  # noqa: E402

BITS = '3.0'
KEPT = Path(__file__).with_name('kept.json')
DTYPES = {np.dtype('int16'): 'I16', np.dtype('int32'): 'I32', np.dtype('float16'): 'F16'}


def gpu_hessians(x, gate, up):
    """recipe/exl3_experts.py's hessians() in float32 on the GPU, where float64 on the CPU would take hours a layer."""
    import torch
    x, gate, up = (torch.from_numpy(a).cuda() for a in (x, gate, up))
    g = x @ gate.T
    a = torch.nn.functional.silu(g) * (x @ up.T)
    return (x.T @ x / len(x)).double().cpu().numpy(), (a.T @ a / len(a)).double().cpu().numpy()


def encode_layer(layer, experts, x, picks, weight, hessians, quantize):
    """The layer's tensors and record: each expert's rows (picks naming it, or all of them), Hessians and projections.
    weight(name) gives a projection's original W; quantize(W, H) gives exllamav3's Q and stored tensors."""
    tensors, record, started = {}, {'rows': {}, 'error': {}}, time.monotonic()
    for expert in experts:
        routed = (picks == expert).any(1)
        rows = int(routed.sum())
        w = {p: weight(ex.name(layer, expert, p)) for p in ex.PROJECTIONS}
        h_in, h_down = hessians(x[routed] if rows else x, w['gate_proj'], w['up_proj'])
        record['rows'][expert], record['error'][expert] = rows, {}
        for p in ex.PROJECTIONS:
            h = h_down if p == 'down_proj' else h_in
            q, stored = quantize(w[p], h)
            record['error'][expert][p] = ex.hessian_error(w[p], q, h)
            tensors.update({f'{ex.name(layer, expert, p)}.{role}': (DTYPES[t.dtype], list(t.shape), t.tobytes())
                            for role, t in stored.items()})
    record['seconds'] = time.monotonic() - started
    return tensors, record


def finished_layer(out, layer, experts):
    """A committed record must match both its requested experts and its shard; never skip a broken copy."""
    done = out / f'layer-{layer:02d}.json'
    if not done.exists():
        return False
    record = json.loads(done.read_text())
    shard = done.with_suffix('.safetensors')
    expected = {str(e) for e in experts}
    if (set(record['rows']) != expected or set(record['error']) != expected
            or any(set(errors) != set(ex.PROJECTIONS) for errors in record['error'].values())
            or not shard.is_file() or ex.sha256(shard) != record['sha256']):
        raise ValueError(f'{done}: incomplete or damaged layer; restore its finished pair before resuming')
    return True


def encode(out, kept, layer_inputs, weight, hessians, quantize):
    """Encodes every layer of KEPT ({layer: experts}) without a record in OUT; layer_inputs(layer) gives its x and
    picks. The record goes last, so a layer stopped mid-write is encoded again."""
    for layer, experts in kept.items():
        done = out / f'layer-{layer:02d}.json'
        if finished_layer(out, layer, experts):
            continue
        tensors, record = encode_layer(layer, experts, *layer_inputs(layer), weight, hessians, quantize)
        ex.write_safetensors(out / f'layer-{layer:02d}.safetensors', tensors, {'format': 'exl3', 'bits': BITS})
        record['sha256'] = ex.sha256(out / f'layer-{layer:02d}.safetensors')
        partial = done.with_suffix('.json.partial')
        partial.write_text(json.dumps(record) + '\n')
        partial.rename(done)
        error = sum(sum(e.values()) for e in record['error'].values())
        print(f'layer {layer}: {len(experts)} experts, {record["seconds"]:.0f} s, error {error:.6g}', flush=True)


def kept_experts():
    return {int(layer): experts for layer, experts in json.loads(KEPT.read_text()).items()}


def main(checkpoint, rows, out):
    import exllamav3_job
    manifest = json.loads((rows / 'rows.json').read_text())
    if manifest['identity']['config_sha256'] != ex.sha256(checkpoint / 'config.json'):
        raise ValueError('calibration rows and checkpoint config differ')
    out.mkdir(parents=True, exist_ok=True)
    # Bind resumed layers to the actual row bytes, including BOTTOM-local captures, not a Mac-only checksum list.
    recipe = {'config': manifest['identity']['config_sha256'],
              'index': ex.sha256(checkpoint / 'model.safetensors.index.json'), 'kept': ex.sha256(KEPT),
              'rows': ex.rows_sha256(rows),
              'encoder': ex.sha256(Path(__file__)), 'quantizer': ex.sha256(Path(exllamav3_job.__file__)),
              'setup': ex.sha256(Path(exllamav3_job.__file__).with_name('exllamav3-env.sh'))}
    stamp = out / 'recipe.json'
    if stamp.exists():
        if json.loads(stamp.read_text()) != recipe:
            raise ValueError(f'{out}: another encoder recipe; use a separate output directory')
    elif any(out.glob('layer-*.json')):
        raise ValueError(f'{out}: finished layers lack recipe.json')
    else:
        partial = stamp.with_suffix('.json.partial')
        partial.write_text(json.dumps(recipe) + '\n')
        partial.rename(stamp)
    quantize_exl3, source = exllamav3_job.import_quantize(), ce.Tensors(checkpoint, np)
    count = manifest['rows']

    def layer_inputs(layer):
        stem, picks, _ = ce.layer_rows(rows, layer, count, np)
        return ce.bf16(f'{stem}.x.bf16', count, np), picks

    encode(out, kept_experts(), layer_inputs, lambda name: ce.original(source, name, np), gpu_hessians,
           lambda w, h: exllamav3_job.quantize(quantize_exl3, w, h, BITS))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
