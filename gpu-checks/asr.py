#!/usr/bin/env python3
"""Transcribe wavs with the pilot's Whisper code, read-only (GPU_CHECK_ASR_DIR transcribe.py, its own venv).

Usage: asr.py MANIFEST OUT_JSON, where MANIFEST holds {'asr_dir': DIR, 'jobs': [{'id', 'wav'}]}. One
whisper-large-v3 pipeline serves every job, cutting at the pilot's own pause detection; OUT_JSON records the
model, the device and each transcript. Nothing under asr_dir is written.
"""
import json
import sys
from pathlib import Path


def main():
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    job = json.loads(manifest.read_text())
    sys.path.insert(0, str(Path(job['asr_dir']).resolve()))
    import librosa  # noqa: E402
    import soundfile as sf  # noqa: E402
    import torch  # noqa: E402
    import transcribe  # noqa: E402
    from transformers import pipeline  # noqa: E402

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipeline('automatic-speech-recognition', model=str(transcribe.snapshot(transcribe.LARGE)), device=device)
    results = {'model': transcribe.LARGE, 'device': device, 'transcripts': {}}
    for row in job['jobs']:
        x, rate = sf.read(row['wav'])
        y = librosa.resample(x.astype('float32'), orig_sr=rate, target_sr=16000)
        spans = transcribe.pieces(len(x) / rate, transcribe.source_pauses(x))
        words = transcribe.transcribe(pipe, y, spans, None)
        results['transcripts'][row['id']] = ' '.join(w['word'] for w in words)
    out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
