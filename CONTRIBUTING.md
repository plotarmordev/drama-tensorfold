# Contributing

Read the [README](README.md), [`recipe/README.md`](recipe/README.md) and
[`engine/patches/README.md`](engine/patches/README.md) first, then fill in the pull request template.

## What belongs here

- **Here:** the Drama 3 patch series in `engine/patches/` and the TensorFold commit it applies to
  (`engine/UPSTREAM`), the launcher in `recipe/`, `scripts/build-engine.sh`, the `bench/` and `qualification/`
  tools, the measured `results/`, and their docs.
- **Upstream [TensorFold](https://github.com/ashhart/TensorFold):** changes that don't depend on Drama 3, such as
  the core, the CUDA runtime, the server outside the speech route and other model families. Send them there,
  under its own contributing guide.
- **Nowhere in git:** model weights, the codec checkpoint, audio, voice clips and voice files (`.gitignore` lists
  them). The weights are under Fish Audio's license, not this repository's.

Moving `engine/UPSTREAM` to another TensorFold commit is a pull request of its own, with the whole series
re-applied and every check below rerun.

Every patch in `engine/patches/` is also proposed upstream: the README's patch table links its TensorFold pull
request. The recipe may ship a patch before TensorFold merges it; a patch without an upstream pull request does not
belong in the series.

## Setup

You need what the README's [Quickstart](README.md#quickstart) lists: CUDA 13, Zig 0.17.0, user systemd and the
memory and disk it states. Then:

```sh
hf download fishaudio/drama-3 --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 --local-dir drama-3
(cd drama-3 && sha256sum -c ../recipe/checksums/checkpoint.sha256)
ZIG=/path/to/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh
```

`scripts/build-engine.sh` checks TensorFold out at the pinned commit in `build/tensorfold`, applies every patch
with `git am` (one commit per patch), builds `tensorfold-native` and `drama3-generate`, and prints
`engine: <upstream> + N patches = <engine HEAD>`. The same series gives the same engine HEAD on every machine.
`recipe/pilot.py` serves the build as the Quickstart shows. The qualification gates also need the reference
runtime directory: see [`qualification/README.md`](qualification/README.md#requirements).

## Changing the engine

Commit your change in `build/tensorfold`, one commit per patch, then export the new commits to the end of the
series and check that the whole series still applies and builds from a clean checkout:

```sh
cd build/tensorfold
git format-patch --start-number 1 -o ../../engine/patches HEAD~2   # two new commits; the v1.1.9 series is empty, so 0001 is next
cd ../../engine/patches && sha256sum -- *.patch > SHA256SUMS
cd ../.. && ZIG=/path/to/zig-0.17.0/zig NVCC=/usr/local/cuda/bin/nvcc scripts/build-engine.sh
```

Export before you rerun `scripts/build-engine.sh`: it checks out the pinned commit again. Add a row to the table in
`engine/patches/README.md`. Change an existing patch only to correct that patch, and keep its file name.

## Tests

Every behaviour change carries a test in the same patch, and the test fails without the change: a threshold test,
for example, fails just above and just below the threshold it pins. In `build/tensorfold`:

```sh
/path/to/zig-0.17.0/zig build test -Dnvcc=/usr/local/cuda/bin/nvcc   # host-side unit tests, no GPU
python3 tools/zig/test_speech_flags.py                                # needs a built tensorfold-native
python3 tools/zig/test_drama_prefill_compare.py
```

A change to `bench/` keeps `python3 bench/stream_check.py --self-test` passing. GPU checks are the gates below,
not unit tests.

## Qualification gates

- **Audio should not change** (speed, loading, refactoring): run the audio-identity gate from
  [`bench/README.md`](bench/README.md#checking-a-speed-change-for-identical-audio) against the baseline build and
  your build. `compare_gate.py` must report `"identical": true` and `stream_check.py` must pass.
- **Numerics change on purpose** (kernels, attention, precision, sampling): rerun the backbone sequence and the
  tower check from [`qualification/README.md`](qualification/README.md) on both builds and report every gate's
  value before and after, passing or not.
- **Thresholds** are declared in the scripts before any run. Never relax one to make a run pass. A bar changes only
  in a pull request of its own, derived from reference data without consulting native results, as the worst-row
  bar was.
- **New results** go into `results/` as JSON with a row in `results/README.md`, and any README number cites its
  file.

## Code standards

- One change per patch, small and focused, and one topic per pull request. A pull request stacked on another says
  which merges first.
- Match the style of the file you change. Comments say why, not what.
- No dead code, commented-out code, debug output, temporary files or leftover scaffolding.
- Tests are deterministic: fixed inputs and seeds, no wall clock, no network and no GPU in unit tests. Native
  sampling is seeded from the prompt, so the same build gives the same bytes for the same text; a test may rely on
  that.
- Patch subjects read `area: what the change does`, with the areas the series uses (`drama:`, `server:`, `tests:`,
  `build:`, `docs:`), and the body says why and what was measured. Commits in this repository say what changed in
  one sentence and name the patches they add, for example "A voice file can hold several reference passages".
- Docs state only what was measured, with the hardware and corpus it was measured on.

## Performance and quality claims

Every speed or quality claim carries before/after evidence from the recipe's bench and qualification scripts,
against the recipe baseline: first-audio time, total time, RTF, WER, speaker cosine and duration ratio. A speed
claim also carries speech evidence from [The Grill](https://github.com/plotarmordev/thegrill), the shared benchmark
harness. The Grill doesn't measure WER or speaker similarity yet, so a quality claim rests on the
[quality](#quality) evidence below alone. Pull requests without a claim need nothing extra. If the numbers don't
support the claim, drop the claim; the pull request can still be opened.

**What counts as a claim:** saying the change makes the recipe faster or its audio better (in the title,
description, a commit message, the changelog or the docs), changing a number in the README or `results/`, or
changing a default because it is faster or better. State the claimed metric and statistic in the pull request
before you measure, for example "warm whole-reply RTF, all-attempts p50", and, when audio changes, the margins
the quality metrics below are judged by.

### Baseline and runs

The baseline is `main` at your branch point (`git merge-base origin/main HEAD`), built by
`scripts/build-engine.sh` from that checkout and served by `recipe/pilot.py` with no serve flags: the built-in
`default` voice, 1024 frames, the pinned checkpoint verified, nothing else on the GPU. A claim about a serve flag
or a voice uses it on every run and says so. Serve flags follow `--` on the pilot's command line; the bench always
asks for the `default` voice, so a voice claim serves a model-invented voice with
`-- --voices "$ROOT/voices" --default-voice NAME`.

Take three runs, each from a fresh `recipe/pilot.py` start:

| Run | Build | Settings |
|---|---|---|
| A | `main` at your branch point | baseline |
| B | your pull request | baseline |
| A2 | the same `main` commit again | baseline |

For a new opt-in flag, all three runs use your pull request's build: A and A2 without the flag, B with it.

### Neutral paths

The pilot and bench records keep every path they are given, the argv, the server's process maps and `HOME`, which
`recipe/pilot.py` passes to the server. Measure under a root that names no user or host, in real directories
rather than symlinks (the scripts resolve paths), with `HOME` inside it. Every command below runs in shells set up
like this:

```bash
export ROOT=/srv/drama HOME=/srv/drama/home   # any root without your user or host name
export CHECKPOINT=$ROOT/drama-3 RUNTIME_DIR=$ROOT/runtime EVIDENCE=$ROOT/evidence GPU_LOCK=/path/to/the/shared/lock
```

Put the checkpoint in `$CHECKPOINT` and verify it, create `$HOME`, `$RUNTIME_DIR` and `$EVIDENCE`, clone your branch
to `$ROOT/pr`, add the baseline next to it with
`git -C "$ROOT/pr" worktree add "$ROOT/base" "$(git -C "$ROOT/pr" merge-base origin/main HEAD)"`, and build both
with `scripts/build-engine.sh`.

### The Grill

Install `grill-perf` [v0.11.0](https://github.com/plotarmordev/thegrill/releases/tag/v0.11.0) in `$ROOT` as its
[INSTALL.md](https://github.com/plotarmordev/thegrill/blob/v0.11.0/docs/performance/INSTALL.md#obtain-verify-unpack)
shows, with `VERSION=0.11.0` and
`ARTIFACT_BASE_URL=https://github.com/plotarmordev/thegrill/releases/download/v0.11.0`: it verifies the checksum
before unpacking and sets `GRILL_PERF`, which the second terminal below needs too.

Then, from `$ROOT/base`, write the runner and the workload every run shares, after the Grill's
[speech servers](https://github.com/plotarmordev/thegrill/blob/v0.11.0/docs/performance/MEDIA.md#speech-servers)
example. The runner streams PCM from the pilot's endpoint and has no `voice` field, so requests get the `default`
voice, the built-in one at baseline, as the bench's do. The workload takes the first short, medium and long texts of
the pinned corpus and gates each on first-audio time and duration ratio; a total-time claim sets `timed` to
`'["first_audio_us", "end_to_end_us"]'`. Declare the gates here, before A, and never change them after it.

```bash
mkdir "$EVIDENCE/grill"
cat > "$EVIDENCE/grill/runner.json" <<'EOF'
{"kind": "media-speech-server-v1", "version": 1, "id": "drama",
 "endpoint": "http://127.0.0.1:18793/v1/audio/speech",
 "model": "fishaudio/drama-3", "response_format": "pcm", "stream": true,
 "pcm": {"channels": 1, "sample_bytes": 2}}
EOF
jq -n --slurpfile corpus bench/corpus/drama-bench-en-v1.jsonl --argjson timed '["first_audio_us"]' '
  [$corpus[] | select(.id | IN("short-01", "medium-01", "long-01"))
   | {id, warmup_runs: 2, measured_runs: 10, request: {prompt: .text, seeds: [1], sample_rate: 44100}}] as $cells
  | {kind: "media-workload-v1", version: 1, name: "drama-speech", modality: "audio",
     request: $cells[0].request, outputs: [], cells: $cells,
     slots: [{id: "a", arm: "a"}, {id: "b", arm: "b"}, {id: "a2", arm: "a2"}],
     limits: {run_deadline_ms: 60000, setup_deadline_ms: 1, output_bytes: 67108864, stdout_bytes: 1, stderr_bytes: 1},
     gates: [$cells[].id as $cell
       | ($timed[] | {cell: $cell, metric: ., max_adverse_bps: 500, max_reference_spread_bps: 2000}),
         {cell: $cell, metric: "duration_ratio", max_adverse_bps: 1000, max_reference_spread_bps: 2000}]}' \
  > "$EVIDENCE/grill/workload.json"
"$GRILL_PERF" media preflight --workload "$EVIDENCE/grill/workload.json" --runner "$EVIDENCE/grill/runner.json"
```

### Measuring

Each run gets a new name: A, B and A2, and a repeated attempt gets another, such as `B-2`, and stays in the
evidence. Set the same `RUN` and `ENGINE` in both shells. In the first, from the run's checkout (`$ROOT/base` for A
and A2, `$ROOT/pr` for B):

```bash
RUN=$EVIDENCE/A ENGINE=$ROOT/base/build/tensorfold
mkdir "$RUN"
python3 recipe/pilot.py --native-bin "$ENGINE/zig-out/native/bin/tensorfold-native" --checkpoint "$CHECKPOINT" \
  --runtime-dir "$RUNTIME_DIR" --gpu-lock "$GPU_LOCK" --evidence-dir "$RUN/pilot"
```

When it prints `DRAMA_PILOT_READY`, in the second, always from `$ROOT/base` so one harness measures every run, with
one `PRECISION` statement for all three and `SLOT` set to `a`, `b` or `a2` (a repeated attempt keeps its run's
slot):

```bash
cd "$ROOT/base" && RUN=$EVIDENCE/A ENGINE=$ROOT/base/build/tensorfold SLOT=a
PID=$(systemctl --user show --property=MainPID --value \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["unit"])' "$RUN/pilot/ready.json")")
COMMON=(--endpoint http://127.0.0.1:18793 --model fishaudio/drama-3 --corpus bench/corpus/drama-bench-en-v1.jsonl
  --precision "$PRECISION" --server-pid "$PID" --server-source "$ENGINE" --checkpoint "$CHECKPOINT"
  --checkpoint-pins recipe/checksums/checkpoint.sha256 --pilot-evidence "$RUN/pilot")
python3 bench/drama_bench.py run "${COMMON[@]}" --evidence-dir "$RUN/whole" --label "${RUN##*/} whole" \
  --server-start fresh
python3 bench/drama_bench.py run "${COMMON[@]}" --evidence-dir "$RUN/stream" --label "${RUN##*/} stream" \
  --server-start reused --response-format pcm --stream
python3 bench/corpus_capture.py http://127.0.0.1:18793 "$RUN/capture" --passes 2
TEXT=$(head -1 bench/corpus/drama-bench-en-v1.jsonl | python3 -c 'import json,sys;print(json.load(sys.stdin)["text"])')
python3 bench/stream_check.py --base http://127.0.0.1:18793 --model fishaudio/drama-3 --input "$TEXT" \
  --out "$RUN/stream-check.json"
jq --arg runtime "$(git -C "$ENGINE" rev-parse HEAD)" \
  --arg hardware "$(nvidia-smi --id=0 --query-gpu=name,driver_version --format=csv,noheader)" \
  '{model_revision: "fishaudio/drama-3@e0a9218aa48888270f06ef1ed9ea67622c0e17d2", $runtime, $hardware,
    settings: (.server_command[3:] | join(" "))}' "$RUN/pilot/ready.json" > "$RUN/grill-deployment.json"
"$GRILL_PERF" media capture --workload "$EVIDENCE/grill/workload.json" --runner "$EVIDENCE/grill/runner.json" \
  --deployment "$RUN/grill-deployment.json" --slot "$SLOT" --out "$RUN/grill"
```

The Grill captures last, so the bench's first phase meets a freshly started server. Its deployment declaration names
the checkpoint, the engine HEAD, the GPU and driver, and the serve flags, so B differs from A and A2 in its engine or
its flags. Stop the pilot with Ctrl-C before the next run. After the three runs, from `$ROOT/base`:

```bash
mkdir "$EVIDENCE/compare"
for base in A A2; do
  for mode in whole stream; do
    python3 bench/drama_bench.py compare "$EVIDENCE/$base/$mode" "$EVIDENCE/B/$mode" \
      --out "$EVIDENCE/compare/B-vs-$base-$mode.json"
  done
done
for run in A2 B; do
  python3 bench/compare_gate.py "$EVIDENCE/A/capture" "$EVIDENCE/A/capture" "$EVIDENCE/$run/capture" \
    "$EVIDENCE/compare/gate-$run.json"
done
"$GRILL_PERF" media compare --a "$EVIDENCE/A/grill" --b "$EVIDENCE/B/grill" --a2 "$EVIDENCE/A2/grill" \
  > "$EVIDENCE/compare/grill.json"
```

### Speed

- **Statistic:** the warm phase's `all_attempts` p50 and p95 in each run's `summary.json`: `first_frame_s` from
  the stream run for first-audio time, `end_to_end_s` and `realtime_factor` from the whole run for total time and
  RTF. Every failure and timeout stays in the denominator and ranks slower than any success. `ok_only` and the
  matched timings in the comparisons drop failures; never base a claim on them.
- **Failures:** report each run's attempted and succeeded counts, its failures by class and
  `server_process_unchanged`. An incomplete run, or one whose server restarted, is not evidence: take it again
  under a new name. A B with more failed requests than A or A2 has no speed claim.
- **Exit codes are not verdicts:** `drama_bench.py run` exits 1 when any request failed, and `compare` exits 0
  for any complete comparison, failures or not.
- **The Grill:** a speed claim also needs `compare/grill.json` to report `"outcome": "PASS"` (exit 0), which takes
  every declared gate passing: first-audio time and duration ratio in each cell, and total time when claimed.
  `INCONCLUSIVE`, `REGRESSION` or an error leaves no speed claim.
- **The claim stands only if B beats both A and A2 on the declared statistic.** Quote B against A, not a bigger
  number. If you push code after measuring, measure again.

### Quality

Native sampling is seeded from the prompt, so A and A2 give the same bytes: `gate-A2.json` must report
`"identical": true`. The quality margin is therefore one you declare, not one A2 measures.

- **Audio should not change:** `gate-B.json` reports `"identical": true` and every `stream-check.json` passes.
  Identical bytes stand in for the metrics below.
- **Audio changes:** compute each metric the same way for A, B and A2, over the warm-phase replies in
  `whole/bodies/` (`samples.jsonl` gives each body's phase):
  - **WER:** corpus-level, against the corpus texts, with one ASR model.
  - **Speaker cosine:** each reply's embedding against one fixed target for all three runs, the reference clip of
    the model-invented voice served. The built-in voice has no clip; score its replies against A's reply to the
    same item, and report that as preservation only.
  - **Duration ratio:** each item's `audio_s` over A's, from `capture/capture.json`, as a median and range. It
    shows dropped or padded speech, not better or worse audio.
- **Preserved**, for a pull request that claims no quality change: B is no worse than A and A2 by more than the
  declared margins on WER and speaker cosine, and its duration ratio is within the declared margin of 1.
- **Improved**, for a quality claim: B beats both A and A2 by at least the declared margin, on WER, on speaker
  cosine against a fixed clip, or on the qualification gates, rerun before and after.

No repository script computes WER or speaker cosine: name the ASR and speaker-embedding models with their
revisions, and give the exact commands. The results here used Whisper and `speechbrain/spkrec-ecapa-voxceleb`.

### Evidence

State the hardware: GPU, OS, driver, CUDA and Zig versions, the checkpoint revision, the recipe commits, the
`"$GRILL_PERF" --version` line, and the `engine: … = <engine HEAD>` line and binary hashes
`scripts/build-engine.sh` printed for each build. Give every command exactly as run. Check the evidence for your
user name, host name and home directory, then archive it without the audio bodies:

```bash
grep -r -l -a -F --exclude-dir=bodies --exclude='*.pcm' --exclude=body.bin -e "$(id -un)" -e "$(hostname)" \
  -e "/home/" "$EVIDENCE"
tar czf evidence.tgz -C "$ROOT" --exclude=bodies --exclude='*.pcm' --exclude=body.bin evidence
```

Never edit evidence. An attempt with a file the check lists is withheld: move it out of `$EVIDENCE`, keep it
privately, never attach it, and measure that attempt again in full under neutral paths. Repeat until the check
prints nothing. The pull request states how many attempts were withheld for privacy and confirms each was
re-measured in full. Every other attempt, failed ones included, stays in the attached archive, which holds
`compare/grill.json` and each run's `grill` capture directory. Keep the bodies, the Grill's `body.bin` files
included, until the pull request is merged. Maintainers check that each comparison's `source_head` is the engine
HEAD of the build it claims to be. Hardware and run conditions are your word, so maintainers may re-run a claim.

## Privacy

- No personal voices, recordings or private names in fixtures, tests, docs, results, commit messages or pull
  request text. Use synthetic voices (model-invented, numbered as in `results/`) or the built-in `default` voice
  only.
- Never commit a clip or voice file, even of your own voice; `clips/` and `voices/` stay git-ignored.
- No host names, user names or private paths either. Evidence is published unedited, so it is collected under
  [neutral paths](#neutral-paths) and checked before it is attached.

## License

All code here is Apache-2.0 ([LICENSE](LICENSE)), including the engine patches and the codec files they create;
[NOTICE](NOTICE) has the attributions. Under section 5 of that license, anything you submit for inclusion is
licensed under Apache-2.0, with no additional terms. Code adapted from elsewhere must have a compatible license,
keep its notice, and be listed in NOTICE with its license text under `LICENSES/`, as flash-linear-attention is.
The codec stays an independent implementation: copy no Fish Audio code into it.

## Review

- Open the pull request against `main` from a branch, and fill in every section of the template.
- There is no CI: the template's Testing section lists the checks you ran, their output, and any you didn't run
  and why.
- A maintainer reviews every patch, test, doc and evidence file, and merges with a merge commit once every finding
  is resolved.
