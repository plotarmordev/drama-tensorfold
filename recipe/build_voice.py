#!/usr/bin/env python3
"""Build a Drama 3 HTTP voice catalog file from encoded reference clips.

Usage:
  drama3-generate --codec drama-3/codec.pth --encode out.codes --ref-audio clip.wav
  python3 recipe/build_voice.py --source synthetic NAME NOTE TRANSCRIPT:CODES [TRANSCRIPT:CODES ...]

Writes runtime/voices/NAME.json in the drama3-http-voice/1 format the patched server
loads with --voices. Transcripts must match their clips; they are the conditioning text.
"""
import argparse
import json
import re
import struct
from pathlib import Path

# Match zig/src/server/drama_voice.zig and drama_prompt.zig in the pinned engine.
MAX_FRAMES = (30 * 44100 + 2047) // 2048
MAX_FILE_BYTES = 256 * 1024
WHITE_SPACE = '\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'
STRUCTURAL_TOKENS = ('<|im_start|>', '<|im_end|>', '<|voice|>', '<|text|>', '<|interleave|>', '<|semantic:')


def read_codes(path: Path) -> list[list[int]]:
    with path.open('rb') as source:
        raw = source.read(MAX_FRAMES * 40 + 1)
    if not raw or len(raw) > MAX_FRAMES * 40 or len(raw) % 40:
        raise ValueError(f'{path}: expected 1..{MAX_FRAMES} complete rows of ten little-endian int32 codes')
    rows = [list(row) for row in struct.iter_unpack('<10i', raw)]
    for row in rows:
        if not 0 <= row[0] < 4096 or any(not 0 <= v < 1024 for v in row[1:]):
            raise ValueError(f'{path}: code out of range')
    return rows


def checked_text(value: str, label: str, transcript: bool = False) -> str:
    if len(value.encode('utf-8')) > 4096 or not value.strip(' \t\r\n'):
        raise ValueError(f'{label}: expected nonblank UTF-8 text of at most 4096 bytes')
    if transcript:
        if not value.lstrip(WHITE_SPACE):
            raise ValueError(f'{label}: transcript must contain text')
        if any(token in value for token in STRUCTURAL_TOKENS):
            raise ValueError(f'{label}: transcript contains a structural prompt token')
        for match in re.finditer(r'<\|speaker', value):
            if not re.match(r'<\|speaker:(?:[0-9]|1[0-5])\|>', value[match.start():]):
                raise ValueError(f'{label}: expected a canonical speaker tag from <|speaker:0|> through <|speaker:15|>')
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, choices=('synthetic', 'self', 'consented'),
                        help='Truthful provenance of every supplied reference')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'runtime/voices')
    parser.add_argument('name')
    parser.add_argument('note', help='Source/consent attestation, not a substitute for permission')
    parser.add_argument('references', nargs='+', metavar='TRANSCRIPT:CODES')
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.name) or args.name.lower() == 'default':
            raise ValueError('name must begin with an ASCII letter/digit, contain only letters/digits/-/_, be at most 64 bytes, and not be default')
        note = checked_text(args.note, 'note')
        if not 1 <= len(args.references) <= 4:
            raise ValueError('expected 1..4 reference clips')
        references = []
        for pair in args.references:
            transcript, separator, codes = pair.rpartition(':')
            if not separator or not codes:
                raise ValueError('expected TRANSCRIPT:CODES for every reference')
            references.append({'text': checked_text(transcript, 'reference', transcript=True),
                               'codes': read_codes(Path(codes))})
        doc = {'format': 'drama3-http-voice/1',
               'source': {'kind': args.source, 'note': note}, 'references': references}
        encoded = (json.dumps(doc, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
        if len(encoded) > MAX_FILE_BYTES:
            raise ValueError(f'catalog exceeds the server limit of {MAX_FILE_BYTES} bytes')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target = args.output_dir / f'{args.name}.json'
        # Do not replace an existing speaker catalog accidentally.
        with target.open('xb') as output:
            output.write(encoded)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'{target}: {len(encoded)} bytes, {len(references)} reference(s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
