### Fixed

- `recipe/speech_server.py`: a long voice note is one take. Segments are now up to 600 characters (was 500),
  `bin/voice-note`'s cap, so a 400-600 character note is no longer two separately drawn takes whose join
  restarted the intonation (a 543-character note had been cut 398 + 143, its second take 20 Hz higher and 5 dB
  louder). Inputs over 600 characters are still segmented as before.
  Listen: `/Users/Shared/agent-work/drama-assets/voice-checks-20261115-one-take/` holds that note's text
  (`text.txt`), `before-two-takes.m4a` (the live voice note, 398 + 143) and `after-one-take.m4a` (the same text
  through this server, one take), both Capri's voice; not in Git, as voice audio stays out.
