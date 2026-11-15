### Changed
- The 14.5 GB int3 build is the default and the recommendation on a Mac and on a GB10: `recipe/pilot.py`,
  `recipe/pilot_macos.py` and `recipe/speech_server.py` without `weights` serve `int3-5900`
  (`weight_selection.DEFAULT_WEIGHTS`), and `scripts/build-engine.sh` without `--weights` builds its `nvfp4` engine
  on both platforms (was `full` on Linux). The README's tip, badges, Quickstart and coding-agent prompt name it first;
  `--weights nvfp4` and `--weights full` stay available.
- The repository's text speaks in the first person singular.

### Added
- CUDA patch `0007-drama-read-BF16-codec-checkpoints.patch`: the CUDA engine reads the int3 build's BF16 `codec.pth`,
  widening each value exactly to FP32, so a GB10 serves the published revision as downloaded. Read through it, all
  535 float tensors equal the original FP32 codec rounded to BF16.
- `results/smaller-builds/hub-card.md`: the Hugging Face card's next README.md, with the GB10 served from the BF16 codec.

### Fixed
- `recipe/README.md` names `pilot_macos.py`'s shared default and lists `weight_selection.py`, `speech_server.py`,
  `pilot_macos.py` and the int3 manifest; `speech_server.py` with no `weights` line serves the default build.
