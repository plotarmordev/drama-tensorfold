#!/usr/bin/env python3
"""The smaller builds of t5900-cover on the Mac: speak the 24 corpus texts, score them, pick the winner.

  measure.py speak RUNS ARM CKPT ref|builtin SALT [SALT ...]   one build in one voice: RUNS/ARM-VOICE/sN/*.wav
  measure.py score RUNS WHISPER_DIR                            per-take WER and failure: RUNS/scores.json
  measure.py report RUNS BASE CKPT_ROOT                        builds.json in this folder, with the winner
  measure.py listen RUNS OUT_HTML                              a self-contained listening page (not uploaded)

Same protocol and failure rule as bench/full-vs-quant (GB10): --max-frames 1024, the eq-3ref reference voice with
guidance 1.5 batched or the built-in voice, Whisper large-v3 with its English normalizer; failed = capped (1024
frames) or WER > 0.2 or a quiet run of 3 s or more inside speech.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from pick import winner

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / 'build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native'
CORPUS = [json.loads(line) for line in (REPO / 'bench/corpus/drama-bench-en-v1.jsonl').read_text().splitlines()]
VOICES = Path(os.environ.get('DRAMA_VOICES', '/tmp/b37/voices'))  # holds eq-3ref.json (sha256 86ccbe54...)
PORT = 18797
GUIDED = ['--voices', str(VOICES), '--default-voice', 'eq-3ref', '--cfg-gamma', '1.5', '--cfg-batched']
LISTEN = ('short-03', 'medium-04', 'long-02')  # one item of each length on the page


def full_vs_quant_score():
    spec = importlib.util.spec_from_file_location('fvq_score', REPO / 'bench/full-vs-quant/score.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_ready(server, log):
    deadline = time.time() + 900
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2)
            return
        except OSError:
            if server.poll() is not None or time.time() > deadline:
                sys.exit(f'server did not start: {log}')
            time.sleep(1)


def speak(runs, arm, ckpt, voice, *salts):
    flags = ['--name', 'fishaudio/drama-3', '--max-frames', '1024'] + (GUIDED if voice == 'ref' else [])
    for salt in salts:
        out = Path(runs) / f'{arm}-{voice}' / f's{salt}'
        out.mkdir(parents=True, exist_ok=True)
        log = out / 'server.log'
        server = subprocess.Popen([str(BIN), 'serve', ckpt, '--host', '127.0.0.1', '--port', str(PORT), *flags],
                                  stdout=log.open('w'), stderr=subprocess.STDOUT,
                                  env={**os.environ, 'TENSORFOLD_SEED_SALT': salt, 'TENSORFOLD_NO_UPDATE_CHECK': '1'})
        try:
            wait_ready(server, log)
            takes = []
            for item in CORPUS:
                body = json.dumps({'model': 'fishaudio/drama-3', 'input': item['text'], 'voice': 'default',
                                   'response_format': 'wav'}).encode()
                request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/audio/speech', body,
                                                 {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=900) as reply:
                    (out / f'{item["id"]}.wav').write_bytes(reply.read())
                    takes.append({'item': item['id'], 'finish': reply.headers.get('X-Finish-Reason')})
        finally:
            server.terminate()
            server.wait(60)
        (out / 'takes.json').write_text(json.dumps(takes, indent=1) + '\n')
        print(out, flush=True)


def score(runs, whisper):
    """Adds the takes not yet in RUNS/scores.json, so a new build costs only its own takes."""
    import soundfile
    import torch
    from transformers import pipeline

    fvq = full_vs_quant_score()
    texts = {item['id']: item['text'] for item in CORPUS}
    asr = pipeline('automatic-speech-recognition', model=whisper, device='mps', torch_dtype=torch.float16)
    normalize = asr.tokenizer.normalize
    path = Path(runs) / 'scores.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r['arm'], r['voice'], r['salt'], r['item']) for r in rows}
    for takes in sorted(Path(runs).glob('*/s*/takes.json')):
        arm, voice = takes.parent.parent.name.rsplit('-', 1)
        salt = int(takes.parent.name[1:])
        for take in json.loads(takes.read_text()):
            if (arm, voice, salt, take['item']) in done:
                continue
            y, rate = soundfile.read(takes.parent / f'{take["item"]}.wav', dtype='float32')
            hyp = asr({'raw': y, 'sampling_rate': rate}, return_timestamps=len(y) > 30 * rate,
                      generate_kwargs={'language': 'english', 'task': 'transcribe'})['text']
            ref, got = normalize(texts[take['item']]).split(), normalize(hyp).split()
            wer = fvq.edit_distance(ref, got) / len(ref)
            quiet = fvq.quiet_run(y, rate)
            capped = len(y) // fvq.FRAME_SAMPLES >= fvq.CAP_FRAMES
            rows.append({'arm': arm, 'voice': voice, 'salt': salt, 'item': take['item'],
                         'finish': take['finish'], 'wer': round(wer, 5), 'ref_words': len(ref),
                         'hypothesis': hyp.strip(), 'quiet_run_s': quiet, 'capped': capped,
                         'failed': capped or wer > fvq.WER_COLLAPSE or quiet >= fvq.QUIET_RUN_S})
    path.write_text(json.dumps(rows, indent=1) + '\n')


def checkpoint_bytes(ckpt):
    """Weights and codec: every *.safetensors and codec.pth, as quant-15gb counts them."""
    return sum(p.stat().st_size for p in Path(ckpt).iterdir() if p.suffix == '.safetensors' or p.name == 'codec.pth')


def report(runs, base, ckpt_root):
    rows = json.loads((Path(runs) / 'scores.json').read_text())
    builds = {}
    for arm in sorted({r['arm'] for r in rows}):
        voices = {}
        for voice in sorted({r['voice'] for r in rows if r['arm'] == arm}):
            takes = [r for r in rows if (r['arm'], r['voice']) == (arm, voice)]
            voices[voice] = {'takes': len(takes), 'wer_mean': round(sum(r['wer'] for r in takes) / len(takes), 5),
                             'wer_pooled': round(sum(r['wer'] * r['ref_words'] for r in takes)
                                                 / sum(r['ref_words'] for r in takes), 5),
                             'failed': sum(r['failed'] for r in takes),
                             'failed_items': sorted(f'{r["item"]} s{r["salt"]}' for r in takes if r['failed'])}
        builds[arm] = {'bytes': checkpoint_bytes(Path(ckpt_root) / arm), 'voices': voices}
    out = {'base': base, 'winner': winner(builds, base), 'builds': builds}
    (HERE / 'builds.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({name: [b['bytes'] / 2**30] + [(v['wer_mean'], v['failed']) for v in b['voices'].values()]
                      for name, b in builds.items()}), 'winner', out['winner'])


def listen(runs, page):
    """Each corpus item in LISTEN, every build and voice side by side, as AAC data URIs in one HTML file."""
    import base64
    import html
    import tempfile

    picked = json.loads((HERE / 'builds.json').read_text())
    builds = sorted(picked['builds'], key=lambda n: -picked['builds'][n]['bytes'])
    scores = {(r['arm'], r['voice'], r['item']): r for r in json.loads((Path(runs) / 'scores.json').read_text())
              if r['salt'] == 0}
    texts = {item['id']: item['text'] for item in CORPUS}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for voice in ('ref', 'builtin'):
            for item in LISTEN:
                cells = []
                for arm in builds:
                    m4a = Path(tmp) / f'{arm}-{voice}-{item}.m4a'
                    subprocess.run(['afconvert', '-f', 'm4af', '-d', 'aac', '-b', '64000',
                                    str(Path(runs) / f'{arm}-{voice}/s0/{item}.wav'), str(m4a)], check=True)
                    s = scores[(arm, voice, item)]
                    cells.append(f'<td><audio controls preload="none" src="data:audio/mp4;base64,'
                                 f'{base64.b64encode(m4a.read_bytes()).decode()}"></audio><br>'
                                 f'WER {100 * s["wer"]:.1f} %{" · failed" if s["failed"] else ""}</td>')
                rows.append(f'<tr><th>{voice}<br>{item}<br><small>{html.escape(texts[item])}</small></th>'
                            + ''.join(cells) + '</tr>')
    head = ''.join(f'<th>{arm}{" (winner)" if arm == picked["winner"] else ""}<br>'
                   f'{picked["builds"][arm]["bytes"] / 2**30:.2f} GiB</th>' for arm in builds)
    Path(page).write_text(
        '<!doctype html><meta charset="utf-8"><title>Drama 3 smaller builds</title>'
        '<style>body{font:14px system-ui;margin:1em}table{border-collapse:collapse}td,th{border:1px solid #ccc;'
        'padding:6px;vertical-align:top;text-align:left}th small{font-weight:normal}audio{width:220px}</style>'
        f'<h1>Drama 3 smaller builds, salt 0</h1><p>Winner: <b>{picked["winner"]}</b>. Not uploaded.</p>'
        f'<table><tr><th></th>{head}</tr>{"".join(rows)}</table>\n')


if __name__ == '__main__':
    {'speak': speak, 'score': score, 'report': report, 'listen': listen}[sys.argv[1]](*sys.argv[2:])
