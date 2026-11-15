#!/usr/bin/env python3
"""Production native Drama backbone vs the pinned SGLang-Omni AR engine, same text fixture.

Native: tf-drama-backbone-oracle runs as a subprocess first and exits (freeing its GPU memory).
Reference: Drama3EngineBuilder.build() makes the real OmniScheduler; one StagePayload carrying a
Drama3GenerationInput (max_new_tokens=1) goes through the real request builder and scheduler loop on
this thread. Drama3ForCausalLM.backbone_logits is wrapped on the instance: the original method runs and
its outputs (final-norm hidden of every row, the logits SGLang keeps) are cloned. No arithmetic is
reimplemented. Run the full mode only inside run-bounded-oracle.py's bounded, full-lifetime GPU guard.
--prepare-only writes the fixture and never initializes CUDA.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REFERENCE_COMMIT = "ab095ef9e779e56af33b6ed2c863645c9903372d"
CHECKPOINT_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
HIDDEN = 2048
CANDIDATES = 4097
CHUNKS = (1, 7, 128)
CONTEXT = 512
MAX_RUNNING_REQUESTS = 1
MAX_NEW_TOKENS = 1
# Bounds this process's PyTorch caching allocator; the parent's external cgroup/admission guard still applies.
REFERENCE_ALLOCATOR_CAP_BYTES = 46 << 30
REQUEST_ID = "backbone-reference-0"
# Ordinary, non-private public text; the default voice tags it <|speaker:0|>.
TEXT = (
    "The old lighthouse on the northern point has guided ships through fog and storm for more than a century. "
    "Every evening at dusk, the keeper climbed the spiral stairs, trimmed the wick, and polished the great glass "
    "lens until it shone. Fishermen returning late with their nets could see the steady beam sweeping across the "
    "dark water, and they knew that the harbor was close. Today the light runs automatically, but visitors still "
    "walk the narrow path along the cliffs to watch the waves break against the rocks below. In spring, wildflowers "
    "cover the hillside, and seabirds build their nests on the ledges beneath the tower. A small museum near the gate "
    "displays old maps, brass instruments, and a logbook filled with careful notes about the weather, the tides, and "
    "the passing ships."
)
# Declared before any run; never relax them to make a run pass.
THRESHOLDS = {
    "hidden_min_cosine_all_rows": 0.999,
    "hidden_max_relative_l2_all_rows": 0.05,
    "hidden_min_row_cosine": 0.995,
    "hidden_min_last_row_cosine": 0.999,
    "semantic_top1_equal": True,
    "semantic_min_top30_overlap": 0.9,
    "semantic_max_kl_reference_to_native": 0.01,
    "semantic_min_cosine": 0.999,
    "semantic_max_relative_l2": 0.03,
}
SCOPE = ("Full 40-layer backbone prefill of one text-only prompt: final-norm hidden for every row and the last "
         "row's EOS-first 4097 semantic logits. Native chunk sizes 1/7/128 against one reference extend. Not decode "
         "parity, tower parity, sampling parity or audio quality.")


class EvidenceError(RuntimeError):
    """The evidence cannot support a comparison; the run fails closed."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=repr) + "\n")


# ---------------------------------------------------------------- Zig prompt contract


def zig_string(literal: str) -> str:
    body, out, i = literal[1:-1], bytearray(), 0
    simple = {"n": b"\n", "r": b"\r", "t": b"\t", "\\": b"\\", '"': b'"', "'": b"'"}
    while i < len(body):
        if body[i] != "\\":
            out += body[i].encode()
            i += 1
        elif body[i + 1] in simple:
            out += simple[body[i + 1]]
            i += 2
        elif body[i + 1] == "x":
            out.append(int(body[i + 2:i + 4], 16))
            i += 4
        elif body[i + 1] == "u" and body[i + 2] == "{":
            end = body.index("}", i)
            out += chr(int(body[i + 3:end], 16)).encode()
            i = end + 1
        else:
            raise EvidenceError(f"unsupported Zig escape in {literal!r}")
    return out.decode()


def tuple_items(source: str) -> list[tuple[str, str]]:
    items, current, depth, i = [], "", 0, 0
    while i < len(source):
        c = source[i]
        if c == '"':
            j = i + 1
            while source[j] != '"':
                j += 2 if source[j] == "\\" else 1
            items.append(("string", zig_string(source[i:j + 1])))
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            if current.strip():
                items.append(("expression", current.strip()))
            current = ""
        elif not (c.isspace() and not current):
            current += c
        i += 1
    if current.strip():
        items.append(("expression", current.strip()))
    return items


def zig_prompt_contract(path: Path) -> dict:
    source = path.read_text()
    constants = {m.group(1): zig_string(m.group(2))
                 for m in re.finditer(r'pub const (\w+) = ("(?:[^"\\]|\\.)*");', source)}
    numbers = {m.group(1): m.group(2) for m in re.finditer(r"pub const (\w+)(?:: \w+)? = ([0-9.]+);", source)}
    body = re.search(r"pub fn segments\([^)]*\)[^{]*\{\s*return \.\{(.*?)\};\s*\}", source, re.S)
    tagged = re.search(r"pub fn tagged\([^)]*\)[^{]*\{(.*?)\n\}", source, re.S)
    if body is None or tagged is None or '"<|speaker:0|> "' not in tagged.group(1):
        raise EvidenceError("drama_prompt.zig segments()/tagged() shape changed")
    segments = []
    for kind, value in tuple_items(body.group(1)):
        if kind == "string":
            segments.append(value)
        elif value in constants:
            segments.append(constants[value])
        elif value == "try tagged(a, input)":
            segments.append(None)
        else:
            raise EvidenceError(f"drama_prompt.zig segment expression {value!r} is not understood")
    if len(segments) != int(numbers.get("segment_count", -1)) or len(segments) != 7 or segments.count(None) != 1:
        raise EvidenceError(f"drama_prompt.zig must give seven segments with one caller slot; got {segments!r}")
    return {"segments": segments, "tts_instruction": constants["tts_instruction"],
            "message_end": constants["message_end"], "generation_start": constants["generation_start"],
            "temperature": float(numbers["temperature"]), "top_p": float(numbers["top_p"]),
            "top_k": int(numbers["top_k"])}


# ---------------------------------------------------------------- fixture


def git_pin(reference: Path) -> dict:
    head = subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if head != REFERENCE_COMMIT or dirty:
        raise EvidenceError(f"reference checkout is {head} (tracked changes: {dirty!r}); expected clean {REFERENCE_COMMIT}")
    return {"commit": head, "tracked_changes": False}


def prepare(args, out: Path) -> tuple[dict, list[int]]:
    sys.path.insert(0, str(args.reference))
    import torch
    from tokenizers import Tokenizer

    import sglang_omni
    from sglang_omni.models.fishaudio_drama3 import prompt as reference_prompt

    if not Path(sglang_omni.__file__).resolve().is_relative_to(args.reference.resolve()):
        raise EvidenceError(f"sglang_omni imported from {sglang_omni.__file__}, not {args.reference}")
    pin = git_pin(args.reference)
    config = json.loads((args.checkpoint / "config.json").read_text())
    contract = zig_prompt_contract(args.drama_prompt)
    if (contract["tts_instruction"], contract["message_end"], contract["generation_start"]) != (
            reference_prompt.TTS_INSTRUCTION, reference_prompt.MESSAGE_END, reference_prompt.GENERATION_START):
        raise EvidenceError("drama_prompt.zig literals differ from the pinned reference prompt.py")
    tagged = reference_prompt.with_speaker_tag(TEXT, 0)
    if tagged != "<|speaker:0|> " + TEXT:
        raise EvidenceError("fixture text unexpectedly carries a leading speaker tag")
    segments = [tagged if s is None else s for s in contract["segments"]]
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    segment_ids = [tokenizer.encode(s).ids for s in segments]
    tokens = [t for ids in segment_ids for t in ids]
    decoder = config["audio_decoder_config"]
    rows = reference_prompt.build_prompt_rows(TEXT, [], tokenizer, num_codebooks=decoder["num_codebooks"],
                                              codebook_size=decoder["vocab_size"])
    if rows.dtype != torch.int64 or rows.shape != (len(tokens), 1 + decoder["num_codebooks"]) \
            or rows[:, 0].tolist() != tokens or bool(rows[:, 1:].any()):
        raise EvidenceError("reference build_prompt_rows differs from the seven separately tokenized Zig segments")
    start, end, vocab = config["semantic_start_token_id"], config["semantic_end_token_id"], config["text_config"]["vocab_size"]
    if not 128 < len(tokens) <= CONTEXT - MAX_NEW_TOKENS:
        raise EvidenceError(f"fixture has {len(tokens)} tokens; need >128 and <= {CONTEXT - MAX_NEW_TOKENS}")
    if any(start <= t <= end or t >= vocab for t in tokens):
        raise EvidenceError("fixture contains semantic or out-of-vocabulary tokens")
    np.asarray(tokens, dtype="<u4").tofile(out / "tokens.u32le")
    fixture = {
        "text": TEXT, "segments": segments, "segment_token_counts": [len(ids) for ids in segment_ids],
        "tokens": len(tokens), "tokens_sha256": sha256(out / "tokens.u32le"),
        "sampling": {k: contract[k] for k in ("temperature", "top_p", "top_k")},
        "reference": {**pin, "path": str(args.reference)},
        "checkpoint": {"path": str(args.checkpoint), "revision_recorded": CHECKPOINT_REVISION,
                       "config_sha256": sha256(args.checkpoint / "config.json"),
                       "tokenizer_sha256": sha256(args.checkpoint / "tokenizer.json"),
                       "index_sha256": sha256(args.checkpoint / "model.safetensors.index.json")},
        "sources_sha256": {"drama_prompt.zig": sha256(args.drama_prompt),
                           "prompt.py": sha256(Path(reference_prompt.__file__)),
                           "backbone-reference-check.py": sha256(Path(__file__))},
        "eos": config["eos_token_id"], "semantic_start": start, "semantic_end": end,
        "thresholds": THRESHOLDS, "scope": SCOPE,
    }
    write_json(out / "fixture.json", fixture)
    if torch.cuda.is_initialized():
        raise EvidenceError("CUDA initialized during fixture preparation")
    return fixture, tokens


# ---------------------------------------------------------------- native


def run_native(args, out: Path, fixture: dict, record: dict) -> None:
    native_dir = out / "native"
    native_dir.mkdir()
    record.update(binary=str(args.native), binary_sha256=sha256(args.native),
                  oracle_source_sha256=sha256(args.native_source))
    serving = args.native.parent.parent.parent / "native/bin/tensorfold-native"
    if serving.is_file():
        record["serving_binary_sha256"] = sha256(serving)
    started = time.monotonic()
    with (out / "native.log").open("w") as log:
        try:
            result = subprocess.run([str(args.native), str(args.checkpoint), str(out / "tokens.u32le"), str(native_dir)],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=args.native_timeout)
            record["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = None
    record["elapsed_s"] = time.monotonic() - started
    record["log"] = (out / "native.log").read_text()[-4000:]
    metadata_path = native_dir / "native-metadata.json"
    if record["returncode"] not in (0, 1) or not metadata_path.is_file():
        raise EvidenceError(f"native oracle did not finish (returncode {record['returncode']}); see native.log")
    metadata = json.loads(metadata_path.read_text())
    expected = {"rows": fixture["tokens"], "hidden_width": HIDDEN, "eos": fixture["eos"],
                "semantic_start": fixture["semantic_start"], "semantic_end": fixture["semantic_end"]}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise EvidenceError(f"native metadata {metadata} disagrees with the fixture {expected}")
    if (record["returncode"] == 0) != (metadata["chunk_bit_exact"] and metadata["reset_bit_exact"]):
        raise EvidenceError("native exit status and its invariant flags disagree")
    record["metadata"] = metadata
    record["files"] = {}
    for chunk in CHUNKS:
        for kind, size in (("hidden.bf16", fixture["tokens"] * HIDDEN * 2), ("semantic.f32", CANDIDATES * 4)):
            path = native_dir / f"native-chunk-{chunk}-{kind}"
            if not path.is_file() or path.stat().st_size != size:
                raise EvidenceError(f"{path.name} missing or not {size} bytes")
            record["files"][path.name] = sha256(path)


# ---------------------------------------------------------------- reference


class LoopWatchdog(threading.Thread):
    """Stops the scheduler loop once the captured request has a terminal message, or at the deadline.

    It only reads the outbox under its lock and clears ``running``; it never touches CUDA.
    """

    def __init__(self, scheduler, deadline_s: float) -> None:
        super().__init__(name="backbone-reference-watchdog", daemon=True)
        self.scheduler, self.deadline_s, self.reason = scheduler, deadline_s, None
        self.finish = threading.Event()

    def run(self) -> None:
        end = time.monotonic() + self.deadline_s
        while not self.finish.is_set():
            outbox = self.scheduler.outbox
            with outbox.mutex:
                terminal = [m.type for m in outbox.queue if m.request_id == REQUEST_ID and m.type in ("result", "error")]
            if terminal or time.monotonic() > end:
                self.reason = terminal[0] if terminal else "timeout"
                self.scheduler.running = False
                return
            time.sleep(0.01)


def plain(value: object) -> object:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return repr(value)


def class_name(value: object) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__qualname__}"


def describe_runtime(scheduler, runner, model) -> dict:
    """Backends and quantization the engine actually resolved; read after the request, so nothing is perturbed."""
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.layers.deep_gemm_wrapper import configurer as deep_gemm_configurer
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    view = resolved_view(scheduler.server_args)
    keys = ("attention_backend", "prefill_attention_backend", "decode_attention_backend", "linear_attn_backend",
            "linear_attn_prefill_backend", "linear_attn_decode_backend", "moe_runner_backend", "fp8_gemm_runner_backend",
            "quantization", "dtype", "kv_cache_dtype", "page_size", "context_length", "max_total_tokens",
            "max_prefill_tokens", "chunked_prefill_size", "max_running_requests", "mem_fraction_static",
            "max_mamba_cache_size", "mamba_ssm_dtype", "disable_cuda_graph", "disable_radix_cache",
            "disable_overlap_schedule", "enable_torch_compile", "enable_fp32_lm_head", "sampling_backend",
            "enable_deterministic_inference", "random_seed", "load_format", "cuda_graph_config")
    experts = next((m for m in model.modules() if isinstance(m, FusedMoE)), None)
    quant = getattr(experts, "quant_method", None)
    attention = runner.attn_backend
    logits_processor = model.logits_processor
    return {
        "resolved_settings": {k: plain(getattr(view, k, "<absent>")) for k in keys},
        "runtime_objects": {
            "model": class_name(model), "attention_backend": class_name(attention),
            "full_attention_backend": class_name(getattr(attention, "full_attn_backend", None)),
            "linear_attention_backend": class_name(getattr(attention, "linear_attn_backend", None)),
            "token_to_kv_pool": class_name(runner.token_to_kv_pool),
            "req_to_token_pool": class_name(runner.req_to_token_pool),
            "req_to_token_pool_size": runner.req_to_token_pool.size,
            "max_total_num_tokens": runner.max_total_num_tokens,
            "prefill_runner": class_name(getattr(runner, "prefill_cuda_graph_runner", None)),
            "decode_cuda_graph_runner": class_name(getattr(runner, "decode_cuda_graph_runner", None)),
            "first_frame_graphs": class_name(model.first_frame_graphs),
            "moe_quant_method": class_name(quant), "moe_quant_runner": class_name(getattr(quant, "runner", None)),
            "moe_runner_backend": repr(get_moe_runner_backend()),
            "deep_gemm_enabled": bool(deep_gemm_configurer.ENABLE_JIT_DEEPGEMM),
            "deep_gemm_scale_ue8m0": bool(deep_gemm_configurer.DEEPGEMM_SCALE_UE8M0),
            "lm_head_weight_dtype": str(model.lm_head.weight.dtype),
            "lm_head_quant_method": class_name(getattr(model.lm_head, "quant_method", None)),
            "logits_fp32_lm_head": bool(logits_processor.use_fp32_lm_head),
            "logit_scale": logits_processor.logit_scale,
            "final_logit_softcapping": logits_processor.final_logit_softcapping,
            "tower_sampler": class_name(model.audio_decoder.sampler),
            "tower_fused_layers": model.audio_decoder.fused_layers is not None,
        },
    }


def run_reference(args, out: Path, fixture: dict, tokens: list[int], record: dict) -> dict:
    import torch

    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "reference.log"), logging.StreamHandler(sys.stderr)])
    main_thread = threading.get_ident()
    record["main_thread"] = main_thread
    total = torch.cuda.get_device_properties(0).total_memory  # first CUDA use: this thread owns the context
    torch.cuda.set_per_process_memory_fraction(min(1.0, REFERENCE_ALLOCATOR_CAP_BYTES / total), 0)
    record["device"] = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                        "total_memory_bytes": total, "allocator_cap_bytes": REFERENCE_ALLOCATOR_CAP_BYTES,
                        "mem_get_info_before": list(torch.cuda.mem_get_info(0))}

    import sglang_omni
    from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
    from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM

    for module in (sglang_omni, sys.modules[Drama3ForCausalLM.__module__]):
        if not Path(module.__file__).resolve().is_relative_to(args.reference.resolve()):
            raise EvidenceError(f"{module.__name__} imported from {module.__file__}, not the pinned checkout")
    record["versions"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "sglang": importlib.metadata.version("sglang"), "sglang_omni": sglang_omni.__version__,
                          "triton": importlib.metadata.version("triton")}
    record["environment"] = {k: v for k, v in os.environ.items()
                             if k.startswith(("SGLANG", "CUDA", "TRITON", "PYTORCH", "NCCL")) or k == "PYTHONPATH"}
    factory = {"context_length": CONTEXT, "max_running_requests": MAX_RUNNING_REQUESTS,
               "max_prompt_rows": CONTEXT - MAX_NEW_TOKENS, "max_new_tokens_cap": MAX_NEW_TOKENS,
               "first_frame_cuda_graphs": False, "first_frame_cuda_graph_max_requests": 1, "enable_async_decode": False}
    overrides = {"disable_cuda_graph": True, "max_total_tokens": CONTEXT, "max_prefill_tokens": CONTEXT}
    record["factory_arguments"], record["server_args_overrides"] = factory, overrides
    started = time.monotonic()
    # One-element holder: teardown pops it so no frame here keeps the engine alive.
    engine = [Drama3EngineBuilder(**factory).build(str(args.checkpoint), device="cuda", gpu_id=0, dtype="bfloat16",
                                                   server_args_overrides=overrides)]
    record["build_s"] = time.monotonic() - started
    try:
        return capture_request(args, out, fixture, tokens, record, engine[0], main_thread)
    finally:
        teardown(engine, record)


def teardown(engine: list, record: dict) -> None:
    """Stop the scheduler, drop the engine and release CUDA memory, all on the CUDA-owning thread."""
    import torch

    errors = []
    scheduler = engine.pop()
    try:
        scheduler.stop()
    except Exception as exc:
        errors.append(f"stop: {type(exc).__name__}: {exc}")
    del scheduler
    gc.collect()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        record.setdefault("memory", {})["mem_get_info_after_teardown"] = list(torch.cuda.mem_get_info(0))
    except Exception as exc:
        errors.append(f"cuda: {type(exc).__name__}: {exc}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as exc:
            errors.append(f"destroy_process_group: {type(exc).__name__}: {exc}")
    record["teardown_errors"] = errors


def capture_request(args, out: Path, fixture: dict, tokens: list[int], record: dict, scheduler, main_thread: int) -> dict:
    import torch

    from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationInput, Drama3GenerationResult
    from sglang_omni.models.fishaudio_drama3.sglang_model import Drama3ForCausalLM
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.scheduling.message import IncomingMessage

    runner = scheduler.tp_worker.model_runner
    model = runner.model
    if type(model) is not Drama3ForCausalLM or model.backbone_logits.__func__ is not Drama3ForCausalLM.backbone_logits:
        raise EvidenceError(f"engine model is {class_name(model)}, not the pinned Drama3ForCausalLM")
    write_json(out / "reference-server-args.json", plain(scheduler.server_args.resolved_dict()))
    candidates = model.audio_decoder.main_token_candidate_ids.cpu().tolist()
    if candidates != [fixture["eos"], *range(fixture["semantic_start"], fixture["semantic_end"] + 1)]:
        raise EvidenceError("reference main_token_candidate_ids are not EOS followed by the semantic band")

    captures = []
    original = model.backbone_logits

    def capturing_backbone_logits(input_ids, positions, forward_batch, frames):
        hidden_states, logits_output = original(input_ids, positions, forward_batch, frames)
        logits = logits_output.next_token_logits
        # Clones are stream-ordered, so later reuse of SGLang's buffers cannot change them.
        captures.append({
            "thread": threading.get_ident(), "mode": forward_batch.forward_mode.name,
            "rids": list(getattr(forward_batch, "rids", None) or []), "batch_size": forward_batch.batch_size,
            "extend_num_tokens": getattr(forward_batch, "extend_num_tokens", None),
            "input_ids": input_ids.detach().clone(), "positions": positions.detach().clone(),
            "frames": frames.detach().clone(), "hidden": hidden_states.detach().clone(),
            "logits": None if logits is None else logits.detach().clone(),
        })
        return hidden_states, logits_output

    sampling = fixture["sampling"]
    codebooks = model.config.audio_decoder_config.num_codebooks
    prompt_rows = torch.zeros((len(tokens), 1 + codebooks), dtype=torch.int64)
    prompt_rows[:, 0] = torch.tensor(tokens, dtype=torch.int64)
    payload = StagePayload(request_id=REQUEST_ID, request=OmniRequest(inputs=None, params={}), data=Drama3GenerationInput(
        prompt_rows=prompt_rows, max_new_tokens=MAX_NEW_TOKENS, temperature=sampling["temperature"],
        top_k=sampling["top_k"], top_p=sampling["top_p"]))
    model.backbone_logits = capturing_backbone_logits
    scheduler.inbox.put(IncomingMessage(request_id=REQUEST_ID, type="new_request", data=payload))
    watchdog = LoopWatchdog(scheduler, args.reference_loop_timeout)
    started = time.monotonic()
    watchdog.start()
    try:
        scheduler.start()  # the real OmniScheduler loop, on this (CUDA-owning) thread
    finally:
        watchdog.finish.set()
        watchdog.join()
        del model.backbone_logits  # restore the class method
    record["loop_s"] = time.monotonic() - started
    record["loop_stop_reason"] = watchdog.reason
    try:
        record.update(describe_runtime(scheduler, runner, model))
    except Exception as exc:  # reported in results.errors; does not alter the captured tensors
        record["runtime_description_error"] = f"{type(exc).__name__}: {exc}"
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    record["outbox"] = [{"request_id": m.request_id, "type": m.type,
                         "data": repr(m.data) if m.type == "error" else class_name(m.data)} for m in messages]
    results = [m for m in messages if m.request_id == REQUEST_ID and m.type == "result"]
    if watchdog.reason != "result" or len(results) != 1 or any(m.type == "error" for m in messages):
        raise EvidenceError(f"scheduler loop ended with {watchdog.reason}; outbox {record['outbox']}")
    generated = results[0].data.data
    if not isinstance(generated, Drama3GenerationResult) or generated.prompt_row_count != len(tokens):
        raise EvidenceError(f"unexpected terminal payload {class_name(generated)}")
    record["generation"] = {"finish_reason": generated.finish_reason, "frames": list(generated.frames.shape)}

    record["captures"] = [{k: v for k, v in c.items() if not isinstance(v, torch.Tensor)} for c in captures]
    extends = [c for c in captures if c["mode"] == "EXTEND"]
    if len(extends) != 1:
        raise EvidenceError(f"expected one EXTEND backbone_logits call; got modes {[c['mode'] for c in captures]}")
    capture = extends[0]
    rows = len(tokens)
    expected_ids = torch.tensor(tokens, dtype=torch.int64)
    hidden, logits = capture["hidden"], capture["logits"]
    frames = capture["frames"].cpu().long()
    checks = {
        "same_thread": capture["thread"] == main_thread,
        "request_ids": capture["rids"] == [REQUEST_ID] and capture["batch_size"] == 1,
        "extend_rows": capture["extend_num_tokens"] == rows,
        "input_ids": torch.equal(capture["input_ids"].cpu().long(), expected_ids),
        "positions": torch.equal(capture["positions"].cpu().long(), torch.arange(rows)),
        "text_frames": tuple(frames.shape) == (rows, 1 + codebooks) and torch.equal(frames[:, 0], expected_ids)
                       and not bool(frames[:, 1:].any()),
        "hidden_shape": tuple(hidden.shape) == (rows, HIDDEN) and hidden.dtype == torch.bfloat16,
        "logits_shape": logits is not None and logits.dim() == 2 and logits.shape[0] == 1
                        and logits.shape[1] == model.logits_processor.vocab_size and logits.dtype == torch.float32,
    }
    record["evidence_checks"] = checks
    if not all(checks.values()):
        raise EvidenceError(f"reference capture checks failed: {[k for k, v in checks.items() if not v]}")
    # The captured last hidden row must be the lm_head input of the captured logits: rerun SGLang's own
    # LogitsProcessor._compute_lm_head on it. A mismatch invalidates the capture; an error leaves it unverified.
    if model.logits_processor.logit_scale is not None or model.logits_processor.final_logit_softcapping:
        record["lm_head_input_consistency"] = "not applicable: logit scaling or softcapping present"
    else:
        try:
            with torch.inference_mode():
                again = model.logits_processor._compute_lm_head(hidden[-1:], model.lm_head)
            record["lm_head_input_consistency"] = bool(torch.equal(again[:, :logits.shape[1]].float(), logits))
        except Exception as exc:
            record["lm_head_input_consistency"] = f"unverified: {type(exc).__name__}: {exc}"
        if record["lm_head_input_consistency"] is False:
            raise EvidenceError("captured hidden is not the lm_head input of the captured logits")
    torch.cuda.synchronize()
    reference_hidden = hidden.view(torch.int16).cpu().numpy().view(np.uint16).reshape(rows, HIDDEN)
    full_logits = logits.cpu().numpy().reshape(-1)
    if not np.isfinite(bits_to_f32(reference_hidden)).all() or not np.isfinite(full_logits).all():
        raise EvidenceError("reference hidden or logits are not finite")
    record["memory"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "max_reserved_bytes": torch.cuda.max_memory_reserved(0),
                        "mem_get_info_after_request": list(torch.cuda.mem_get_info(0))}
    semantic = full_logits[np.asarray(candidates)]
    reference_hidden.astype("<u2").tofile(out / "reference-hidden.bf16")
    frames.numpy().astype("<u4").tofile(out / "reference-frames.u32le")
    full_logits.astype("<f4").tofile(out / "reference-full-logits.f32")
    semantic.astype("<f4").tofile(out / "reference-semantic.f32")
    record["files"] = {name: sha256(out / name) for name in
                       ("reference-hidden.bf16", "reference-frames.u32le", "reference-full-logits.f32", "reference-semantic.f32")}
    captures.clear()
    return {"hidden_bits": reference_hidden, "semantic": semantic}


# ---------------------------------------------------------------- comparison


def bf16_bits(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype="<u2").reshape(shape)


def bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def round_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint32)
    return (rounded << 16).view(np.float32)


def vector_metrics(native: np.ndarray, reference: np.ndarray) -> dict:
    a, b = native.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise EvidenceError("shape mismatch or non-finite values")
    an, bn, error = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(a - b)
    return {"max_abs": float(np.max(np.abs(a - b))), "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "relative_l2": float(error / bn) if bn else (0.0 if error == 0 else float("inf")),
            "cosine": float(np.dot(a, b) / (an * bn)) if an and bn else float(an == bn),
            "equal_elements": int(np.count_nonzero(native.ravel() == reference.ravel())), "elements": int(a.size)}


def hidden_metrics(native_bits: np.ndarray, reference_bits: np.ndarray) -> dict:
    native, reference = bits_to_f32(native_bits).astype(np.float64), bits_to_f32(reference_bits).astype(np.float64)
    ordered = lambda bits: np.where(bits & 0x8000, -(bits & 0x7FFF).astype(np.int32), (bits & 0x7FFF).astype(np.int32))
    ulp = np.abs(ordered(native_bits) - ordered(reference_bits))
    dots = np.sum(native * reference, axis=1)
    norms = np.linalg.norm(native, axis=1) * np.linalg.norm(reference, axis=1)
    row_cosine = np.where(norms > 0, dots / np.where(norms > 0, norms, 1), 1.0)
    row_relative = np.linalg.norm(native - reference, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    result = {
        "all_rows": vector_metrics(native, reference),
        "last_row": vector_metrics(native[-1], reference[-1]),
        "rows": {"min_cosine": float(row_cosine.min()), "min_cosine_row": int(row_cosine.argmin()),
                 "median_cosine": float(np.median(row_cosine)), "max_relative_l2": float(row_relative.max()),
                 "max_relative_l2_row": int(row_relative.argmax()),
                 "rows_below_min_cosine": int(np.count_nonzero(row_cosine < THRESHOLDS["hidden_min_row_cosine"]))},
        "bf16_ulp": {"bit_equal_fraction": float(np.mean(ulp == 0)), "within_1_fraction": float(np.mean(ulp <= 1)),
                     "within_2_fraction": float(np.mean(ulp <= 2)), "p99": float(np.percentile(ulp, 99)),
                     "max": int(ulp.max())},
    }
    result["pass"] = (result["all_rows"]["cosine"] >= THRESHOLDS["hidden_min_cosine_all_rows"]
                      and result["all_rows"]["relative_l2"] <= THRESHOLDS["hidden_max_relative_l2_all_rows"]
                      and result["rows"]["min_cosine"] >= THRESHOLDS["hidden_min_row_cosine"]
                      and result["last_row"]["cosine"] >= THRESHOLDS["hidden_min_last_row_cosine"])
    return result


def log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    return shifted - np.log(np.exp(shifted).sum())


def candidate_token(index: int, fixture: dict) -> int:
    return fixture["eos"] if index == 0 else fixture["semantic_start"] + index - 1


def semantic_metrics(native: np.ndarray, reference: np.ndarray, fixture: dict) -> dict:
    result = vector_metrics(native, reference)
    a, b = native.astype(np.float64), reference.astype(np.float64)
    a_order, b_order = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
    a_lp, b_lp = log_softmax(a), log_softmax(b)
    result.update(
        top1_equal=bool(a_order[0] == b_order[0]),
        native_top1_token=candidate_token(int(a_order[0]), fixture),
        reference_top1_token=candidate_token(int(b_order[0]), fixture),
        native_rank_of_reference_top1=int(np.flatnonzero(a_order == b_order[0])[0]),
        native_top1_margin=float(a[a_order[0]] - a[a_order[1]]),
        reference_top1_margin=float(b[b_order[0]] - b[b_order[1]]),
        top30_overlap=len(set(a_order[:30].tolist()) & set(b_order[:30].tolist())) / 30,
        kl_reference_to_native=float(np.sum(np.exp(b_lp) * (b_lp - a_lp))),
        kl_native_to_reference=float(np.sum(np.exp(a_lp) * (a_lp - b_lp))),
        reference_eos_probability=float(np.exp(b_lp[0])), native_eos_probability=float(np.exp(a_lp[0])),
        diagnostic_native_rounded_to_bf16=vector_metrics(round_bf16(native), reference),
    )
    result["pass"] = (result["top1_equal"] and result["top30_overlap"] >= THRESHOLDS["semantic_min_top30_overlap"]
                      and result["kl_reference_to_native"] <= THRESHOLDS["semantic_max_kl_reference_to_native"]
                      and result["cosine"] >= THRESHOLDS["semantic_min_cosine"]
                      and result["relative_l2"] <= THRESHOLDS["semantic_max_relative_l2"])
    return result


def compare(out: Path, fixture: dict, reference: dict) -> dict:
    rows = fixture["tokens"]
    comparisons = {}
    for chunk in CHUNKS:
        native_hidden = bf16_bits(out / "native" / f"native-chunk-{chunk}-hidden.bf16", (rows, HIDDEN))
        native_semantic = np.fromfile(out / "native" / f"native-chunk-{chunk}-semantic.f32", dtype="<f4")
        case = {"hidden": hidden_metrics(native_hidden, reference["hidden_bits"]),
                "semantic": semantic_metrics(native_semantic, reference["semantic"], fixture)}
        case["pass"] = case["hidden"]["pass"] and case["semantic"]["pass"]
        comparisons[f"chunk-{chunk}"] = case
        print(json.dumps({"chunk": chunk, "pass": case["pass"],
                          "hidden_cosine": case["hidden"]["all_rows"]["cosine"],
                          "hidden_min_row_cosine": case["hidden"]["rows"]["min_cosine"],
                          "semantic_top1_equal": case["semantic"]["top1_equal"],
                          "semantic_top30_overlap": case["semantic"]["top30_overlap"],
                          "semantic_kl": case["semantic"]["kl_reference_to_native"]}), flush=True)
    return comparisons


LIMITS = [
    "One prompt, one reference extend; no decode-step, tower, sampling or audio parity is measured.",
    "Reference logits are SGLang's bf16 lm_head output cast to f32 unless enable_fp32_lm_head resolved true; "
    "native logits are f32. See semantic.diagnostic_native_rounded_to_bf16.",
    "Reference prefill uses SGLang's extend kernels (chunked gated-delta rule, triton extend attention, fp8 MoE with "
    "dynamic activation quantization); native uses its own kernels. Differences are expected at bf16 noise level; "
    "MoE top-k near-ties can amplify them on individual rows.",
    "The native oracle binary is built from the same backbone sources as the serving binary, but its hash is not "
    "the serving binary's hash; linkage is by source, not proven by this run.",
    "Checkpoint weights are not re-hashed here (cost); verify them against recipe/checksums/checkpoint.sha256.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="sglang-omni checkout at REFERENCE_COMMIT")
    parser.add_argument("--engine", type=Path, required=True, help="patched TensorFold tree with its oracle tools built")
    parser.add_argument("--native-timeout", type=float, default=240.0)
    parser.add_argument("--reference-loop-timeout", type=float, default=120.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.native = args.engine / "zig-out/oracle/bin/tf-drama-backbone-oracle"
    args.native_source = args.engine / "zig/tests/cuda/drama_backbone_oracle.zig"
    args.drama_prompt = args.engine / "zig/src/server/drama_prompt.zig"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture, tokens = prepare(args, out)
    if args.prepare_only:
        import torch
        print(json.dumps({"prepared": True, "cuda_initialized": torch.cuda.is_initialized(), "tokens": fixture["tokens"],
                          "tokens_sha256": fixture["tokens_sha256"], "fixture_sha256": sha256(out / "fixture.json")}))
        return 0
    result = {"schema": "backbone-reference-check/1", "scope": SCOPE, "thresholds": THRESHOLDS,
              "fixture_sha256": sha256(out / "fixture.json"), "tokens_sha256": fixture["tokens_sha256"],
              "evidence_valid": False, "passed": False, "errors": [], "warnings": [], "limits": LIMITS,
              "native": {}, "reference": {}}
    try:
        run_native(args, out, fixture, result["native"])
        reference = run_reference(args, out, fixture, tokens, result["reference"])
        result["comparisons"] = compare(out, fixture, reference)
        result["evidence_valid"] = True
        native_invariants = result["native"]["metadata"]["chunk_bit_exact"] and result["native"]["metadata"]["reset_bit_exact"]
        result["passed"] = native_invariants and all(c["pass"] for c in result["comparisons"].values())
    except EvidenceError as exc:
        result["errors"].append(f"EvidenceError: {exc}")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        logging.getLogger(__name__).exception("backbone reference check failed")
    reference_record = result["reference"]
    if "runtime_description_error" in reference_record:
        result["warnings"].append(f"runtime settings not recorded: {reference_record['runtime_description_error']}")
    if isinstance(reference_record.get("lm_head_input_consistency"), str):
        result["warnings"].append(f"lm_head input consistency {reference_record['lm_head_input_consistency']}")
    result["warnings"] += [f"teardown {e}" for e in reference_record.get("teardown_errors", [])]
    write_json(out / "results.json", plain(result))
    print(json.dumps({"passed": result["passed"], "evidence_valid": result["evidence_valid"], "errors": result["errors"],
                      "results_sha256": sha256(out / "results.json")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
