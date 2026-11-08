"""Pinned checkpoint selection shared by the Linux and macOS launchers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parent
NVFP4_SERVE = ('--cfg-gamma', '1.5', '--cfg-batched')
# 'engine' is the scripts/build-engine.sh --weights profile that builds 'build'; both quants share one.
WEIGHTS = {
    'full': {'source': 'checkpoint', 'dir': 'drama-3', 'pins': 'checkpoint.sha256',
             'engine': 'full', 'build': 'tensorfold', 'serve': ()},
    'nvfp4': {'source': 'nvfp4_checkpoint', 'dir': 'drama-3-NVFP4-Pruned', 'pins': 'nvfp4.sha256',
              'engine': 'nvfp4', 'build': 'tensorfold-nvfp4', 'serve': NVFP4_SERVE},
    'nvfp4-5900': {'source': 'nvfp4_5900_checkpoint', 'dir': 'drama-3-NVFP4-Pruned-5900',
                   'pins': 'nvfp4-t5900cover.sha256',
                   'engine': 'nvfp4', 'build': 'tensorfold-nvfp4', 'serve': NVFP4_SERVE},
}
GUIDANCE_FLAGS = ('--cfg-gamma', '--cfg-allow-request-gamma', '--cfg-batched')


def checkpoint_path(choice: str | None, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    if choice is None:
        raise ValueError('Pass --weights nvfp4 or --weights full, or --checkpoint DIR')
    return (REPO / WEIGHTS[choice]['dir']).resolve()


def native_path(choice: str | None, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    if choice is None:
        raise ValueError('The native backend requires --native-bin, or --weights')
    return (REPO / 'build' / WEIGHTS[choice]['build'] / 'zig-out/native/bin/tensorfold-native').resolve()


def serve_arguments(choice: str | None, arguments: list[str]) -> list[str]:
    selected = list(arguments)
    if choice and not any(a.split('=', 1)[0] in GUIDANCE_FLAGS for a in selected):
        selected[:0] = WEIGHTS[choice]['serve']
    return selected


def native_environment(choice: str | None) -> list[str]:
    # Preserve batched prefill when the pilot clears the inherited environment.
    return [] if choice and WEIGHTS[choice]['engine'] == 'nvfp4' else ['TF_DRAMA_PREFILL_ROWS=128']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pins(pins: Path) -> list[list[str]]:
    return [line.split(None, 1) for line in pins.read_text().splitlines() if line.strip()]


def prepare_checkpoint(choice: str | None, override: Path | None) -> dict:
    """Verify selected pins, downloading only an absent default (never an override).

    Explicit legacy paths have operator-owned provenance, not an implied pinned source.
    """
    target = checkpoint_path(choice, override)
    provenance = {'weights_mode': choice or 'explicit', 'checkpoint_path': str(target),
                  'checkpoint_source': None, 'checkpoint_manifest': None,
                  'checkpoint_verified': False}
    if choice is None:
        return provenance
    spec = WEIGHTS[choice]
    source = json.loads((RECIPE / 'sources.json').read_text())[spec['source']]
    pins = RECIPE / 'checksums' / spec['pins']
    repair = (f'Repair it with: hf download {source["repository"]} --revision {source["revision"]} '
              f'--local-dir {target}, or delete it and run again')
    if override is None and not target.exists() and not target.is_symlink():
        hf = shutil.which('hf')
        if hf is None:
            raise ValueError('the hf command is missing; install it with: python3 -m pip install -U huggingface_hub')
        print(f'Downloading {source["repository"]} at {source["revision"][:8]} into {target}', flush=True)
        subprocess.run([hf, 'download', source['repository'], '--revision', source['revision'],
                        '--local-dir', str(target)], check=True)
    print(f'Checking {target} against {pins.relative_to(REPO)}', flush=True)
    started = time.monotonic()
    bad = [name for digest, name in read_pins(pins)
           if not (target / name).is_file() or sha256(target / name) != digest]
    if bad:
        raise ValueError(f'{target} does not match {pins}: {", ".join(bad)} missing or changed; no model was '
                         f'started. {repair}')
    print(f'{target} matches {pins.relative_to(REPO)} ({time.monotonic() - started:.0f} s)', flush=True)
    provenance.update(checkpoint_source=source, checkpoint_manifest=str(pins), checkpoint_verified=True)
    return provenance
