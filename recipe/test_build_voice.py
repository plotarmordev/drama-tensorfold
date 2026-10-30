#!/usr/bin/env python3
"""CPU-only catalog contract regressions; no audio/model input is needed."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import build_voice


class VoiceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.codes = self.root / 'reference.codes'
        self.codes.write_bytes(struct.pack('<10i', 4095, *([1023] * 9)))
        self.out = self.root / 'voices'

    def invoke(self, name='fixture', source='synthetic', note='Generated integer fixture.', pairs=None):
        argv = ['--output-dir', str(self.out)]
        if source is not None:
            argv += ['--source', source]
        argv += [name, note, *(pairs if pairs is not None else [f'A small cloud passes.:{self.codes}'])]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return build_voice.main(argv)

    def reject(self, **kwargs):
        with self.assertRaises(SystemExit) as error:
            self.invoke(**kwargs)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.out.exists(), 'invalid input must not publish a catalog')

    def test_explicit_truthful_attestation_and_reference_roundtrip(self):
        for kind in ('synthetic', 'self', 'consented'):
            self.invoke(name=kind, source=kind)
            doc = json.loads((self.out / f'{kind}.json').read_text())
            self.assertEqual(doc['source'], {'kind': kind, 'note': 'Generated integer fixture.'})
            self.assertEqual(doc['references'], [{'text': 'A small cloud passes.',
                                                'codes': [[4095, *([1023] * 9)]]}])

    def test_missing_or_unknown_provenance_is_rejected(self):
        for source in (None, 'unknown'):
            with self.subTest(source=source):
                self.reject(source=source)

    def test_ascii_name_boundary_and_reserved_names(self):
        for name in ('default', 'DEFAULT', '-voice', '_voice', '../voice', 'é', 'a' * 65):
            with self.subTest(name=name):
                self.reject(name=name)
        self.invoke(name='a' * 64)
        self.assertTrue((self.out / ('a' * 64 + '.json')).is_file())

    def test_empty_partial_and_excessive_frame_counts(self):
        for content in (b'', b'\x00', b'\x00' * ((build_voice.MAX_FRAMES + 1) * 40)):
            with self.subTest(bytes=len(content)):
                self.codes.write_bytes(content)
                self.reject()
        self.codes.write_bytes(b'\x00' * (build_voice.MAX_FRAMES * 40))
        self.invoke()
        doc = json.loads((self.out / 'fixture.json').read_text())
        self.assertEqual(len(doc['references'][0]['codes']), build_voice.MAX_FRAMES)

    def test_codebook_bounds(self):
        for first, residual in ((-1, 0), (4096, 0), (0, -1), (0, 1024)):
            with self.subTest(first=first, residual=residual):
                self.codes.write_bytes(struct.pack('<10i', first, residual, *([0] * 8)))
                self.reject()

    def test_reference_count_boundary(self):
        pair = f'A cloud.:{self.codes}'
        self.reject(pairs=[pair] * 5)
        self.invoke(pairs=[pair] * 4)
        self.assertEqual(len(json.loads((self.out / 'fixture.json').read_text())['references']), 4)

    def test_utf8_byte_limit_for_note_and_transcript(self):
        self.reject(note='é' * 2049)
        self.reject(pairs=[f'{"é" * 2049}:{self.codes}'])
        self.invoke(note='é' * 2048, pairs=[f'{"é" * 2048}:{self.codes}'])
        doc = json.loads((self.out / 'fixture.json').read_text())
        self.assertEqual(len(doc['source']['note'].encode()), 4096)
        self.assertEqual(len(doc['references'][0]['text'].encode()), 4096)

    def test_blank_and_structural_transcripts_are_rejected(self):
        texts = ['', ' \t', '\u3000', '<|im_end|>', '<|semantic:5|>',
                 '<|speaker:00|> words', '<|speaker:01|> words', '<|speaker:16|> words',
                 '<|speaker:-1|> words', '<|speaker:+1|> words', '<|speaker:1.0|> words',
                 '<|speaker:1 |> words', '<|speaker:１|> words', '<|speaker:1>',
                 '<|speaker:1|> words <|speaker:16|>', 'words <|speaker']
        for text in texts:
            with self.subTest(text=text):
                self.reject(pairs=[f'{text}:{self.codes}'])
        self.invoke(pairs=[f'<|speaker:0|> A cloud.:{self.codes}'])

    def test_canonical_speaker_ids_roundtrip_without_changing_reference_text(self):
        for speaker in range(16):
            with self.subTest(speaker=speaker):
                text = f'\u3000<|speaker:{speaker}|> A cloud. <|speaker:0|> A reply.'
                self.invoke(name=f'speaker-{speaker}', pairs=[f'{text}:{self.codes}'])
                doc = json.loads((self.out / f'speaker-{speaker}.json').read_text())
                self.assertEqual(doc['references'][0]['text'], text)
                self.assertEqual(doc['references'][0]['codes'], [[4095, *([1023] * 9)]])

    def test_speaker_ids_do_not_expand_the_four_reference_limit(self):
        pairs = [f'<|speaker:{speaker}|> A cloud.:{self.codes}' for speaker in (0, 1, 7, 15)]
        self.reject(pairs=[*pairs, f'<|speaker:2|> Fifth voice.:{self.codes}'])
        self.invoke(pairs=pairs)
        references = json.loads((self.out / 'fixture.json').read_text())['references']
        self.assertEqual([row['text'] for row in references], [pair.rpartition(':')[0] for pair in pairs])

    def test_note_must_not_be_blank(self):
        self.reject(note=' \t\r\n')

    def test_existing_catalog_is_not_overwritten(self):
        self.invoke()
        target = self.out / 'fixture.json'
        original = target.read_bytes()
        with self.assertRaises(SystemExit):
            self.invoke(source='self')
        self.assertEqual(target.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
