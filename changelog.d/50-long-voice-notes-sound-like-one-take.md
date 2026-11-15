### Fixed

- `recipe/speech_server.py`: a long voice note is one take. Segments are now up to 600 characters (was 500),
  `bin/voice-note`'s cap, so a 400-600 character note is no longer two separately drawn takes whose join
  restarted the intonation (a 543-character note had been cut 398 + 143, its second take 20 Hz higher and 5 dB
  louder). Inputs over 600 characters are still segmented as before.
