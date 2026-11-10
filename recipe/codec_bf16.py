#!/usr/bin/env python3
"""Convert a Drama 3 codec state dict's FP32 tensors to BF16 without changing other dtypes."""

import argparse
from collections import Counter
from pathlib import Path

import torch


def convert(state):
    counts = Counter()
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('codec checkpoint must be a flat tensor state dict')
        counts[str(tensor.dtype)] += 1
        if tensor.dtype == torch.float32:
            state[name] = tensor.to(torch.bfloat16)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    state = torch.load(args.source, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('codec checkpoint must be a tensor state dict')
    before = convert(state)
    torch.save(state, args.destination)
    saved = torch.load(args.destination, map_location='cpu', weights_only=True)
    after = Counter(str(t.dtype) for t in saved.values())
    print(f'{args.source}: {args.source.stat().st_size} bytes, {dict(before)}')
    print(f'{args.destination}: {args.destination.stat().st_size} bytes, {dict(after)}')


if __name__ == '__main__':
    main()
