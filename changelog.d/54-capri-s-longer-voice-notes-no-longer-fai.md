### Fixed

- `bin/voice-note` (installed as `~/bin/voice-note`) is now in the repo, with tests. A note that runs out of time in
  TOP's `voice` menu (exit 124; deadline 20 s + 0.15 s per character, max 57 s) now says it timed out, not that
  Capri's voice is down, and exits 124. Every other failure still reads as voice down, and no other voice stands in.
  Live check: a 400-character note came back as a 21.5 s m4a in 31 s.
