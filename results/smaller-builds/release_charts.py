#!/usr/bin/env python3
"""The Hugging Face card's charts for the int3 build against t5900, from builds.json and mac-speed.json.

  release_charts.py OUT_DIR

Writes footprint.png (weights and codec on disk, each smaller-build step), quality.png (card 37's word error rate
and failed takes, both voices) and speed.png (warm real-time factor on the M5 Ultra). Needs matplotlib.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
GIB = 2 ** 30
COLOURS = ('#4c72b0', '#dd8452', '#55a868', '#c44e52')
LABELS = {'t5900': 'NVFP4 5,900\n(previous release)', 'trim': 'head trimmed', 'trim-bf16': 'and BF16 codec',
          'trim-bf16-int3': 'and int3 experts\n(this build)'}
PAIR = ('t5900', 'trim-bf16-int3')


def bars(ax, names, values, title, unit, fmt):
    ax.bar(range(len(names)), values, color=COLOURS[:len(names)] if len(names) > 2 else (COLOURS[0], COLOURS[2]))
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(max(values) * 1.18, 1))
    for i, v in enumerate(values):
        ax.annotate(fmt.format(v), (i, v), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points',
                    fontsize=9)


def main(out):
    builds = json.loads((HERE / 'builds.json').read_text())['builds']
    speed = json.loads((HERE / 'mac-speed.json').read_text())['builds']
    out.mkdir(parents=True, exist_ok=True)

    steps = list(LABELS)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars(ax, steps, [builds[n]['bytes'] / GIB for n in steps], 'Weights + codec on disk', 'GiB', '{:.2f}')
    fig.suptitle('Drama 3, 5,900 experts: footprint of each step')
    fig.tight_layout()
    fig.savefig(out / 'footprint.png', dpi=150)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    for row, (voice, name) in zip(axes, (('ref', 'reference voice'), ('builtin', 'built-in voice'))):
        stats = [builds[n]['voices'][voice] for n in PAIR]
        bars(row[0], PAIR, [s['wer_mean'] * 100 for s in stats], f'Word error rate, {name} (mean)', '%', '{:.2f}')
        bars(row[1], PAIR, [s['failed'] for s in stats], f'Failed takes of {stats[0]["takes"]}, {name}', 'takes',
             '{:d}')
    fig.suptitle('Quality on the M5 Ultra: 24 texts × 2 salts per voice')
    fig.tight_layout()
    fig.savefig(out / 'quality.png', dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars(axes[0], PAIR, [speed[n]['rtf_warm']['median'] for n in PAIR], 'Real-time factor, warm, median\n'
         '(lower is faster)', 'RTF', '{:.3f}')
    bars(axes[1], PAIR, [speed[n]['rtf_warm']['total'] for n in PAIR], 'Real-time factor, warm, whole pass',
         'RTF', '{:.3f}')
    fig.suptitle('Speed on the M5 Ultra (Metal): built-in voice, 24-text corpus, one request at a time')
    fig.tight_layout()
    fig.savefig(out / 'speed.png', dpi=150)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
