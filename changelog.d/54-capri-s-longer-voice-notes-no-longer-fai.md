### Fixed

- `bin/voice-note` (installed as `~/bin/voice-note`) is now in the repo, with tests. A note that runs out of time in
  TOP's `voice` menu (exit 124; deadline 20 s + 0.15 s per character, max 57 s) now says it timed out, not that
  Capri's voice is down, and exits 124. A note the menu turns away because another is being spoken (exit 75, busy)
  waits up to a minute for its turn instead of reading as voice down. Every other failure still reads as voice
  down, and no other voice stands in. Notes get unique names (two in the same second no longer overwrite each
  other), are readable only by their user (folder 0700, audio 0600), and leave no temporary file behind.
  Live check: two 400-character notes started together both came back as 21.5 s m4a files, in 22 s and 45 s.
