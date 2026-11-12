# Capri's cleaned long voice stops dropping out

The cleaned long catalog (`capri-clean-long`, 47.5 s across two references, sha256 `201fcd55…`) ends speech
early on some Capri notes: about 3 in 7 lines speak ~0.5 s and then fall silent for 10 s
(`X-Finish-Reason: silence`). The same lines speak fully in `shell-nolaugh` on the same int3 build.

The fix is a shorter reference cut from the same cleaned recording, installed under the same voice name:
`capri-clean-long.json` is now 7.2 s / 156 codec frames, one reference, sha256 `923b561a…`. It is the cut card
43's cleanup recommended ("cleaned short is the clearest noise-reduction choice"), and it keeps Capri's voice:
same source, same cleanup, no other voice. The old 47.5 s catalog is kept as `capri-clean-long-47s.json`
(sha256 `201fcd55…`) for the before rows.

## Results

20 Capri notes of 1–3 sentences (100–400 characters), one request each, on the int3 build
(`trim-bf16-int3`, engine `fb8fdf3d` + the five Mac patches, `--cfg-gamma 1.5 --cfg-batched`,
`--max-frames 4096`, silence guard -40 dBFS / 10 s / 1 redraw). The two lines that dropped out on 12 Nov are
lines 1 and 2.

| Voice | Lines | `stop` | `silence` | Shortest | Longest | Most trailing silence |
|---|---:|---:|---:|---:|---:|---:|
| `capri-clean-long` (fixed) | 20 | 20 | 0 | 5.76 s | 9.71 s | 0.36 s |
| `shell-nolaugh` | 20 | 20 | 0 | 4.46 s | 8.50 s | 0.34 s |
| `capri-clean-long-47s` (before) | 2 | 0 | 2 | 10.36 s | 10.68 s | 10.06 s |

Every fixed-catalog line ends `stop` with at most 0.36 s of trailing silence; the two known lines speak 9.29 s
and 7.62 s. The before rows are the same two lines on the old catalog: both end `silence` after ~0.3–0.7 s of
speech.

Per-line rows: [`results.json`](results.json), which also keeps the fixed catalog's sha256, frame count and seconds.
The catalog itself is not in Git (reference audio, codec rows and generated catalogs stay out); it is deployed at
`/Users/Shared/agent-work/drama-assets/voices/capri/catalogs/capri-clean-long.json` (sha256 `923b561a…`).

The fixed catalog was measured on a second engine of the same int3 build (engine sha256 `5493f1fe…`), because
the live server loads its catalog folder at start; the `shell-nolaugh` rows are from the live server on port
18793. Each row's `server` field names the engine that produced it (`second-engine` or `live-18793`). Both
engines are the same build profile and the same two known lines behave the same on each.

## Files for listening

The 20 WAVs per voice are in `/Users/Shared/agent-work/drama-assets/voice-checks-20261112/`:
`capri-clean-long/01.wav` … `20.wav` and `shell-nolaugh/01.wav` … `20.wav`; the two before takes are in
`capri-clean-long-47s/`. They are not in Git (`*.wav` is ignored); the JSON keeps the measurements.

## Installing

Copy the deployed `/Users/Shared/agent-work/drama-assets/voices/capri/catalogs/capri-clean-long.json` to the
release's voice catalog folder as `capri-clean-long.json` (the voice name stays `capri-clean-long`), keeping the
old catalog as `capri-clean-long-47s.json`. The catalog is
`drama3-http-voice/1`, one reference, 156 frames, transcript "Hello, I'm here. Hi. How you doing? Pretty good,
actually. It's really nice to meet you." — the same cleaned recording, cut at 7.2 s.

## Caveat

The measurements show every line speaks to the end; they do not judge likeness. Net should listen to the WAVs
before the Mac switches Capri back.
