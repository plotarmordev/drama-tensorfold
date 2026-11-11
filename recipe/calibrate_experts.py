#!/usr/bin/env python3
"""Capture Drama 3 per-expert calibration rows on Apple silicon and report their coverage and statistics.

The Metal engine (`drama3-generate --calibration-rows`) generates every request, then prefills its prompt and the
generated frames again and dumps the MoE input x, the eight original-ID picks and their routing weights for every row
of all 40 layers, in the layout of the GPTQ --rows dumps. No CUDA, torch or Spark is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import wave

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path('/Users/Shared/agent-work/drama-assets')
CHECKPOINT = ASSETS / 'checkpoint-bf16'
NVFP4 = ASSETS / 'drama3-nvfp4-t5900cover-v1/model'
OUTPUT = ASSETS / 'exl3-calibration-rows-v2'
VOICE = ASSETS / 'voices/capri'
CORPUS = ROOT / 'bench/corpus/drama-bench-en-v1.jsonl'
BINARY = ROOT / 'build/tensorfold/zig-out/bin/drama3-generate'
LAYERS, EXPERTS, TOP_K, HIDDEN = 40, 256, 8, 2048
CLASSES = ('prompt_text', 'prompt_audio', 'generated')
REFERENCE_SECONDS, REFERENCES, SEEDS = 29, 3, (1, 2)
CHECK_LAYERS = (0, 3, 20, 39)
# These rows come from other text and audio than the NVFP4 build's, so its statistics are matched closely, not exactly.
ERR_H_TOLERANCE = 0.05
# Rows are weighted to the NVFP4 build's class mix; a class that needs more than this factor either way is captured
# too far from that mix for weights to stand in for rows.
MIX_LIMIT = 2
# Kept experts allowed to have no rows (Drama lead, 11 Nov): over the 175,396-row capture none places above 171st of
# the router's 256 logits, and the NVFP4 build's calibration gave them no rows either. Each uses its layer's rows.
NEVER_ROUTED = {36: (0, 14, 39, 117, 120, 147, 173, 195, 202, 211, 227, 246)}
FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
HESSIAN = {
    'gate_up': 'X_e = the rows of layer-L.x.bf16 whose picks contain e, row i weighted by c_i = report.json '
               'class_weights[class.u8[i]]; H = X_e^T diag(c) X_e / sum(c)',
    'down': 'A_e = silu(X_e W_gate^T) * (X_e W_up^T) with the original weights, as one unquantized forward pass '
            'collects them; H = A_e^T diag(c) A_e / sum(c)',
    'no_rows': 'an expert the calibration never routed to takes X_e = all of its layer\'s rows',
    'error': 'trace((W - Q) H (W - Q)^T), W the FP8 original; NVFP4 Q is the published expert, GPTQ-calibrated on '
             'other rows, so its error is measured on this run\'s H too',
}


def references(work, binary=BINARY, checkpoint=CHECKPOINT):
    """Encodes REFERENCES consecutive clips of the Capri recording; each gets the matching share of its transcript."""
    source = work / 'capri.wav'
    subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEI16@44100', '-c', '1', str(VOICE / 'source.mp3'), str(source)], check=True)
    words = (VOICE / 'transcript.txt').read_text().split()
    with wave.open(str(source)) as audio:
        rate, total = audio.getframerate(), audio.getnframes()
        clips = []
        for i in range(REFERENCES):
            clip = work / f'capri-{i}.wav'
            with wave.open(str(clip), 'wb') as out:
                out.setparams(audio.getparams())
                out.writeframes(audio.readframes(REFERENCE_SECONDS * rate))
            subprocess.run([str(binary), '--encode', str(clip.with_suffix('.codes')), '--ref-audio', str(clip),
                            '--codec', str(checkpoint / 'codec.pth')], check=True)
            # The recording is one steady talk, so a clip's words are taken in proportion to its time span.
            span = slice(len(words) * i * REFERENCE_SECONDS * rate // total, len(words) * (i + 1) * REFERENCE_SECONDS * rate // total)
            clips.append({'ref_text': ' '.join(words[span]), 'ref_codes': str(clip.with_suffix('.codes'))})
    return clips


def requests(texts, clips):
    """Every corpus line alone and after every reference clip, at each seed, plus the whole corpus three times over."""
    out = [{'text': text, 'seed': seed} for text in texts for seed in SEEDS]
    out += [{'text': text, 'seed': seed, **clip} for clip in clips for text in texts for seed in SEEDS]
    out.append({'text': '\n\n'.join(texts * 3), 'seed': 1})
    return out


def capture(output, checkpoint=CHECKPOINT):
    work = output.with_name(output.name + '-requests')
    work.mkdir()
    texts = [json.loads(line)['text'] for line in CORPUS.read_text().splitlines() if line.strip()]
    lines = work / 'requests.jsonl'
    lines.write_text(''.join(json.dumps(r) + '\n' for r in requests(texts, references(work, checkpoint=checkpoint))))
    subprocess.run([str(BINARY), '--model', str(checkpoint), '--calibration-rows', str(output), '--text', str(lines),
                    '--max-frames', '1500', '--max-ctx', '8192', '--max-rows', '1024'], cwd=ROOT, check=True)


def kept_experts(index=NVFP4 / 'model.safetensors.index.json'):
    """The (layer, expert) pairs the 5,900-expert NVFP4 checkpoint holds."""
    kept = set()
    for name in json.loads(index.read_text())['weight_map']:
        parts = name.split('.')
        if 'experts' in parts and parts[-1] == 'weight':
            kept.add((int(parts[parts.index('layers') + 1]), int(parts[parts.index('experts') + 1])))
    return kept


def layer_rows(output, layer, rows, np):
    stem = output / f'layer-{layer:02d}'
    picks = np.fromfile(f'{stem}.picks.i32', '<i4').reshape(rows, TOP_K)
    weights = np.fromfile(f'{stem}.w.f32', '<f4').reshape(rows, TOP_K)
    return stem, picks, weights


def widen(raw, np):
    """BF16 bit patterns to float32."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def bf16(path, rows, np):
    return widen(np.fromfile(path, '<u2'), np).reshape(rows, HIDDEN)


def column_squares(x, picks, c, np):
    """Per-expert, per-column sum of c x^2 over the rows routed to the expert: the diagonal of X_e^T diag(c) X_e."""
    squares = x * x * c[:, None]
    return np.stack([squares[(picks == expert).any(1)].sum(0, dtype=np.float64) for expert in range(EXPERTS)])


def class_weights(counts, target):
    """Each class's share of the NVFP4 build's statistics (TARGET, its stats_pairs) over its share of these rows
    (COUNTS): the weighted rows have the NVFP4 build's class mix."""
    rows, pairs = sum(counts.values()), sum(target.values())
    return {name: target[name] / pairs * rows / counts[name] if counts[name] else float('inf') for name in CLASSES}


def best_ranks(x, gate, experts, np):
    """Each expert's best router rank over the rows x, 0 being the top logit; the router keeps ranks 0 to 7. A tie
    ranks the higher expert ID lower, so a tied expert is never reported as reachable when it may not be."""
    logits = x @ gate.T
    ids = np.arange(logits.shape[1])[None, :]
    return {e: int(((logits > logits[:, [e]]) | ((logits == logits[:, [e]]) & (ids < e))).sum(1).min()) for e in experts}


class Tensors:
    """Reads named tensors from a sharded safetensors checkpoint without ML packages."""

    def __init__(self, root, np):
        self.root, self.np = root, np
        self.files = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def __call__(self, name):
        path = self.root / self.files[name]
        if path not in self.headers:
            with path.open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                self.headers[path] = (8 + size, json.loads(f.read(size)))
        start, header = self.headers[path]
        meta = header[name]
        dtype = {'F32': '<f4', 'U8': 'u1', 'F8_E4M3': 'u1', 'BF16': '<u2'}[meta['dtype']]
        begin, end = meta['data_offsets']
        raw = self.np.fromfile(path, dtype, (end - begin) // self.np.dtype(dtype).itemsize, offset=start + begin)
        if meta['dtype'] == 'F8_E4M3':
            raw = e4m3(raw, self.np)
        elif meta['dtype'] == 'BF16':
            raw = widen(raw, self.np)
        return raw.reshape(meta['shape'])


def e4m3(raw, np):
    bits = np.arange(256)
    exponent, mantissa = (bits >> 3) & 15, bits & 7
    value = np.where(exponent == 0, mantissa / 8 * 2.0 ** -6, (1 + mantissa / 8) * 2.0 ** (exponent - 7.0))
    value[(bits & 127) == 127] = np.nan
    return np.where(bits >= 128, -value, value).astype(np.float32)[raw]


def original(tensors, name, np):
    """FP8 weight times its 128x128 block scales."""
    w, scale = tensors(name + '.weight'), tensors(name + '.weight_scale_inv')
    return w * np.repeat(np.repeat(scale, 128, 0), 128, 1)[:w.shape[0], :w.shape[1]]


def nvfp4(tensors, name, np):
    """Two E2M1 values per byte (low nibble first) times the per-16 E4M3 scale and the global scale."""
    packed = tensors(name + '.weight')
    codes = np.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    return np.array(FP4, np.float32)[codes] * np.repeat(tensors(name + '.weight_scale'), 16, 1) * tensors(name + '.weight_scale_2')


def stats_check(output, rows, kept, checkpoint, c, np):
    """err_h of the NVFP4 gate/up experts recomputed with this run's column sums of squares, against nvfp4-provenance.json."""
    published = {t['name']: t['err_h'] for t in json.loads((NVFP4 / 'nvfp4-provenance.json').read_text())['tensors']}
    source, quant = Tensors(checkpoint, np), Tensors(NVFP4, np)
    ratios = []
    for layer in CHECK_LAYERS:
        stem, picks, _ = layer_rows(output, layer, rows, np)
        squares = column_squares(bf16(f'{stem}.x.bf16', rows, np), picks, c, np)
        for expert in sorted(e for l, e in kept if l == layer):
            for projection in ('gate_proj', 'up_proj'):
                name = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}'
                w = original(source, name, np)
                h = squares[expert]
                ours = np.sqrt((h * (w - nvfp4(quant, name, np)) ** 2).sum() / (h * w * w).sum())
                ratios.append(float(ours / published[name + '.weight']))
    ratios = np.array(ratios)
    return {
        'method': 'err_h = sqrt(sum h (W - Q)^2 / sum h W^2) with W the FP8 original, Q the published NVFP4 expert and h '
                  'the per-column sum of c x^2 from these rows, c the class weight; equal h gives a ratio of exactly 1 '
                  'to the published err_h',
        'layers': list(CHECK_LAYERS),
        'tensors': len(ratios),
        'ratio_median': float(np.median(ratios)),
        'ratio_p05_p95': [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))],
        'tolerance': ERR_H_TOLERANCE,
        'why_not_exactly_1': 'the NVFP4 statistics (stats.safetensors, kept on the Spark) come from its own '
                             'calibration text and audio; the class weights match only its class mix',
    }


def nvfp4_routed(provenance=NVFP4 / 'nvfp4-provenance.json'):
    """Rows per (layer, expert) in the NVFP4 build's own calibration, from its GPTQ metadata."""
    return {(source['layer'], expert): count for source in json.loads(provenance.read_text())['layer_sources']
            for expert, count in enumerate(json.loads(source['metadata']['gptq'])['rows'])}


def validate(output, checkpoint=CHECKPOINT, kept=None, reference=None, target=None, weight_checks=True):
    """Checks a capture and reports its coverage and class weights; TARGET is the NVFP4 build's stats_pairs.
    WEIGHT_CHECKS adds the checks that read the checkpoint's weights."""
    import numpy as np

    manifest = json.loads((output / 'rows.json').read_text())
    rows = manifest['rows']
    config_sha = hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest()
    if (manifest['schema'], manifest['layers'], manifest['identity']['config_sha256']) != ('drama-probe-rows/1', LAYERS, config_sha):
        raise ValueError(f'{output}: wrong schema, layer count or checkpoint identity')
    classes = np.fromfile(output / 'class.u8', np.uint8)
    if len(classes) != rows or classes.max() >= len(CLASSES):
        raise ValueError(f'{output}: class.u8 does not hold one valid class per row')
    kept = kept_experts() if kept is None else kept
    reference = nvfp4_routed() if reference is None else reference
    target = json.loads((NVFP4 / 'nvfp4-provenance.json').read_text())['stats_pairs'] if target is None else target
    class_rows = {name: int((classes == i).sum()) for i, name in enumerate(CLASSES)}
    missing, missing_kept, routed = {}, {}, {}
    for layer in range(LAYERS):
        stem, picks, weights = layer_rows(output, layer, rows, np)
        if Path(f'{stem}.x.bf16').stat().st_size != rows * HIDDEN * 2:
            raise ValueError(f'{stem}.x.bf16: wrong size')
        if picks.min() < 0 or picks.max() >= EXPERTS or (np.sort(picks, 1)[:, 1:] == np.sort(picks, 1)[:, :-1]).any():
            raise ValueError(f'layer {layer}: a row does not hold eight distinct original expert IDs')
        if not np.allclose(weights.sum(1), 1, atol=1e-5) or (weights < 0).any():
            raise ValueError(f'layer {layer}: routing weights are not a distribution')
        counts = np.bincount(picks.ravel(), minlength=EXPERTS)
        routed[layer] = counts
        missing[layer] = [int(e) for e in np.flatnonzero(counts == 0)]
        missing_kept[layer] = [e for e in missing[layer] if (layer, e) in kept]
    kept_counts = [int(routed[l][e]) for l, e in kept]
    report = {
        'schema': 'drama-exl3-calibration/2',
        'producer': 'recipe/calibrate_experts.py + TensorFold Metal drama3-generate --calibration-rows',
        'checkpoint': str(checkpoint),
        'config_sha256': config_sha,
        'rows': rows,
        'sequences': len(manifest['chunks']),
        'class_rows': class_rows,
        'nvfp4_stats_pairs': target,
        'class_weights': class_weights(class_rows, target),
        'routed_experts': LAYERS * EXPERTS,
        'covered_experts': sum(EXPERTS - len(m) for m in missing.values()),
        'kept_experts': len(kept),
        'covered_kept_experts': len(kept) - sum(len(m) for m in missing_kept.values()),
        'kept_rows_min_median': [min(kept_counts), int(np.median(kept_counts))],
        'missing_kept_by_layer': {str(l): m for l, m in missing_kept.items() if m},
        'never_routed_exception': {str(l): list(e) for l, e in NEVER_ROUTED.items()},
        'missing_kept_nvfp4_routed': {str(l): [e for e in m if reference[l, e]] for l, m in missing_kept.items()
                                      if any(reference[l, e] for e in m)},
        'missing_by_layer': {str(l): m for l, m in missing.items() if m},
        'bytes': sum(p.stat().st_size for p in output.iterdir() if p.name != 'report.json'),
        'hessian': HESSIAN,
    }
    if weight_checks:
        router = Tensors(checkpoint, np)
        # Why a kept expert got no rows: its best rank among the router's 256 logits over all of its layer's rows.
        report['missing_kept_best_rank'] = {
            str(layer): best_ranks(bf16(output / f'layer-{layer:02d}.x.bf16', rows, np),
                                   router(f'model.language_model.layers.{layer}.mlp.gate.weight'), experts, np)
            for layer, experts in missing_kept.items() if experts}
        c = np.array([report['class_weights'][name] for name in CLASSES])[classes]
        report['stats_check'] = stats_check(output, rows, kept, checkpoint, c, np)
    return report


def failures(report):
    """Why a capture is not usable for the EXL3 quant: a kept expert without rows outside NEVER_ROUTED, a class too far
    from the NVFP4 build's mix or statistics outside the tolerance."""
    out = []
    unexpected = {l: u for l, m in report['missing_kept_by_layer'].items()
                  if (u := [e for e in m if e not in NEVER_ROUTED.get(int(l), ())])}
    if unexpected:
        out.append(f'kept experts without rows: {unexpected}')
    if not all(1 / MIX_LIMIT <= w <= MIX_LIMIT for w in report['class_weights'].values()):
        out.append(f'class weights {report["class_weights"]} outside 1/{MIX_LIMIT}-{MIX_LIMIT}: capture closer to '
                   f'the NVFP4 class mix {report["nvfp4_stats_pairs"]}')
    if not all(abs(r - 1) <= ERR_H_TOLERANCE for r in report['stats_check']['ratio_p05_p95']):
        out.append(f'err_h ratio outside 1 ± {ERR_H_TOLERANCE}: these rows do not match the NVFP4 statistics')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='new directory; an existing capture is only validated')
    args = parser.parse_args()
    if not (args.output / 'rows.json').exists():
        capture(args.output, args.checkpoint)
    report = validate(args.output, args.checkpoint)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rows', 'class_rows', 'class_weights', 'covered_kept_experts', 'kept_experts', 'bytes')}))
    print(f'best router ranks of kept experts without rows: {report["missing_kept_best_rank"]}')
    print(f'err_h ratio to the NVFP4 build: {report["stats_check"]["ratio_p05_p95"]} (5th-95th percentile)')
    if problems := failures(report):
        raise SystemExit('\n'.join(problems))


if __name__ == '__main__':
    main()
