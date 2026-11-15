# Full model against quant 01

Drama 3 full FP8 against the NVFP4 quant (routed experts GPTQ, 7,369 of 10,240 kept), on one GB10 host, one engine
binary for every arm, from run `run-01` of `bench/full-vs-quant/run.py`. Five arms, four seed salts each:

| Arm | Checkpoint | Voice | Guidance |
|---|---|---|---|
| F | full FP8 | reference voice | none |
| Q | quant | reference voice | γ 1.5, batched |
| FG | full FP8 | reference voice | γ 1.5, batched |
| FB | full FP8 | built-in voice | none |
| QB | quant | built-in voice | γ 1.5 requested; the engine logged γ 1.00 (inactive without a reference) |

| File | What it holds |
|---|---|
| [`card-section.md`](card-section.md) | The model-card section: the recipe (quant, engine, codec), the tables, the charts, The Grill's speed comparison and the run history, including the resume |
| [`hub-card.md`](hub-card.md) | The Hugging Face card's README.md for `plotarmordev/drama-3-NVFP4-Pruned`, kept by hand |
| [`tables.md`](tables.md) | The per-arm table and the paired comparisons |
| [`tables.json`](tables.json) | Every number in the tables, the environment, the resume record, the pin change and, under `grill`, The Grill's comparison |
| [`charts/`](charts) | `footprint`, `speed` and `quality`, each as PNG and SVG |
| [`raw/`](raw) | Per-request records, engine log lines, per-take scores (with transcripts of the public bench corpus), speaker-similarity pairs and `grill-compare.json`, The Grill's comparison byte for byte (numbers, hashes and reply headers only) |

`tables.json` SHA-256 `f3c96e49…`, from merged scores SHA-256 `9e13c6a9…` and speaker pairs SHA-256 `88b72ca4…`
(byte-identical to `raw/scores.jsonl` and `raw/speaker-pairs.jsonl`) and The Grill's comparison SHA-256 `ba79d2c9…`
(byte-identical to `raw/grill-compare.json`).

`report.py` writes every file here except this README and `hub-card.md`, which are maintained by hand. A rerun of the
command below with the same matplotlib reproduces the other files byte for byte, the SVGs included (fixed
`svg.hashsalt`, no date). The tables mark which rows are nearest-rank percentiles from the benchmark client's
summaries and which are arithmetic medians computed by `report.py`.

## The Grill

A separate speed comparison of the FG configuration (slots a and a2) against Q (slot b) with The Grill
(`grill-perf 0.11.0`, `media compare`): overall INCONCLUSIVE (exit 2) because the model changes between slots by
design, not because a capture failed; first audio and end-to-end PASS in short-07, medium-01 and long-01, duration
ratio REGRESSION in short-07 and long-01 (the quant's audio 0.821 and 0.876 of the full model's length). It measures
its own three cells and bounds, not output quality, and is not pooled into run-01. Numbers, claim scope and how it sits
beside run-01's rows: `card-section.md`, section "Speed under The Grill". Only the comparison is here; the three
captures hold prompts and audio and stay on the GPU host.

## Run history

The first pass stopped at the quant's first salt: the benchmark client refused to start it because the quant's pin
list held Hugging Face download metadata (`.cache/huggingface/`) and `.gitattributes`, hidden paths the client does
not list. Commit `7f14908` dropped the metadata from the pins and gave the client the same list less hidden paths;
every kept entry has the sha256 it had before. The run resumed from that salt with the full model's four salts kept;
the refused directory was set aside and is in no number. `card-section.md` and `tables.json` carry the counts.

## How it was made

Scoring ran on a CPU host (Whisper large-v3 in float32; the GPU path uses float16), in two processes split so every
speaker pair stays inside one process: the reference-voice arms (F, Q, FG) and the built-in-voice arms (FB, QB).
Their outputs were concatenated. The run directory is a copy of the GPU host's; nothing was re-run.

```sh
mkdir -p runs/score-views/fq runs/score-views/builtin
for a in F Q FG; do ln -s ../../run-01/$a runs/score-views/fq/$a; done
for a in FB QB; do ln -s ../../run-01/$a runs/score-views/builtin/$a; done
for v in fq builtin; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=9 bench/full-vs-quant/score.py runs/score-views/$v \
    --whisper WHISPER_LARGE_V3_DIR --ecapa SPKREC_ECAPA_VOXCELEB_DIR --speechbrain-path SPEECHBRAIN_DIR &
done; wait
cat runs/score-views/{fq,builtin}/scores.jsonl > runs/run-01/scores.jsonl
cat runs/score-views/{fq,builtin}/pairs.jsonl > runs/run-01/pairs.jsonl
bench/full-vs-quant/report.py runs/run-01 results/full-vs-quant-01 \
  --grill GRILL_RESULTS/compare.json --grill-collector 'grill-perf 0.11.0' \
  --grill-collector-sha256 4491eedb76d90215a7990b92419ad98c23667854805a1e175beb7ab105f7981a \
  --grill-arms a=FG,b=Q,a2=FG
```
