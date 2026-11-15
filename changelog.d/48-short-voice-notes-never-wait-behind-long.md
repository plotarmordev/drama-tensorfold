### Changed

- `recipe/speech_server.py` gives the engine's one lane out a segment at a time: a short note (under 400 characters)
  sent during a long reply is spoken before that reply's next segment, so it waits for at most one segment instead of
  the whole reply. Long replies keep the same audio and still speak one at a time. The log records each take's start
  (characters and seconds queued) and end (seconds on the engine and seconds of audio).
- Every speech request holds a shared flock on `/tmp/tensorfold-speech.lock` from arrival to its last byte, so
  TensorFold's batch GPU tools (`tf-h3-image`, `tf-h3-dit`) pause for live speech through a whole reply, gaps between
  segments included, not only while patch 0007's engine is mid-take. See
  [the speech server](recipe/README-macos.md#speech-server).
