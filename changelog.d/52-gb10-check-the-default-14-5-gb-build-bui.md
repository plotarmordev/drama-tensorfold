### Added
- `results/gb10-default/gb10-default.sh`: a GB10 job (`job start gpu top`, no network) that builds the default engine
  with `scripts/build-engine.sh` from a TensorFold copy in the pilot folder, checks the int3 weights against
  `recipe/checksums/trim-bf16-int3.sha256`, serves them without systemd and asks for speech once, printing the weights
  revision, HTTP status, seconds of audio and time to first audio.

### Changed
- `recipe/pilot.py`, `recipe/pilot_macos.py` and the GB10 job build the native server's command line with one
  `weight_selection.server_command`.
