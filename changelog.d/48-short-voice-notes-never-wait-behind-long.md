### Changed

- `recipe/speech_server.py` gives the engine's one lane out a segment at a time: a short note (under 400 characters)
  sent during a long reply is spoken before that reply's next segment, so it waits for at most one segment instead of
  the whole reply. Long replies keep the same audio and still speak one at a time. The log records each take's start
  (characters and seconds queued) and end (seconds on the engine). See
  [the speech server](recipe/README-macos.md#speech-server).
