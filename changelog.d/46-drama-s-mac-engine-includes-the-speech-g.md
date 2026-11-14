### Fixed

- The Mac NVFP4 engine gives way to speech: patch `0007-speech-holds-gpu-turn.patch` carries the speech side of
  plotarmordev/TensorFold #49, so every speech request holds a shared flock on `/tmp/tensorfold-speech.lock` and
  stamps `/tmp/tensorfold-speech.lock.stamp`, and a render on the same Mac pauses until the voice is done. Both
  files open across users. The pin `fb8fdf3d` and the CUDA pins are unchanged; every int3 corpus take is
  byte-identical to the live engine's ([`results/macos-speech-hold.json`](results/macos-speech-hold.json)).
