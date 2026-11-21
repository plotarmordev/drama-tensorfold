#!/usr/bin/env python3
"""Per-frame eos standing and chosen tokens from a drama3-generate logits trace.

The tracer dumps, per frame of a take: the 253k candidate logits (`--logits-out`),
the 11 chosen columns (`--out …frames`: main token, 9 codebook codes, speaker) and
the prompt rows (`--rows-out`). `derive` turns a trace into the compact columnar
table `stop-decision-20261121.json` cites — eos's rank, probability and cumulative
probability, the top candidate's share, whether each truncated nucleus (request
top_p 0.9, RAS top_p 0.95) keeps eos, and the drawn main token — without the 16 MiB
logits. `analyze` additionally replays the kernel's draws (the Philox port below)
to check the table against the recorded frames.

Usage: trace_eos.py derive PREFIX OUT.json | analyze PREFIX SEED
"""
import hashlib
import json
import struct
import sys

import numpy as np

EOS = 248046
SEM_BEGIN = 248099
SEM_END = 252194
NCAND = 1 + SEM_END - SEM_BEGIN + 1
REQUEST = dict(top_k=30, top_p=0.9)  # the request draw
RAS = dict(top_k=30, top_p=0.95)     # the anti-spam redraw
M32 = 0xFFFFFFFF


def philox_first(seed, c0, c1, c2, c3):
    k0 = seed & M32
    k1 = (seed >> 32) & M32
    for _ in range(10):
        _c0, _c2 = c0, c2
        c0 = (umulhi(0xCD9E8D57, _c2) ^ c1 ^ k0) & M32
        c2 = (umulhi(0xD2511F53, _c0) ^ c3 ^ k1) & M32
        c1 = (0xCD9E8D57 * c2) & M32
        c3 = (0xD2511F53 * _c0) & M32
        k0 = (k0 + 0x9E3779B9) & M32
        k1 = (k1 + 0xBB67AE85) & M32
    return c0


def umulhi(a, b):
    return ((a * b) >> 32) & M32


def exp_noise(seed, step, row, stream, index):
    key = (seed ^ ((step >> 32) << 32)) & 0xFFFFFFFFFFFFFFFF
    bits = philox_first(key, index & M32, row & M32, stream & M32, step & M32)
    u = ((bits >> 9) + 0.5) * (1.0 / 8388608.0)
    return -np.log(u)


def draw(logits, temperature, top_k, top_p, seed, step, stream, eos=-1):
    """The kernel's draw: sorted candidates kept by rank 0, the eos slot (patch 0008)
    or the top_k/top_p nucleus, racing weight over Exp(1) noise."""
    order = np.argsort(-logits, kind="stable")
    temp = max(temperature, 1e-5)
    mx = logits[order[0]] / temp
    zt = np.exp(logits / temp - mx).sum()
    best, brank = -1.0, 1 << 30
    cum = 0.0
    for i in range(len(order)):
        idx = int(order[i])
        wv = np.exp(logits[idx] / temp - mx)
        cum += wv
        if not (i == 0 or idx == eos or (i < top_k and cum / zt <= top_p)):
            continue
        race = wv / exp_noise(seed, step, 0, stream, idx)
        if race > best:
            best, brank = race, i
    return int(order[brank])


def seed_for(ids, salt):
    h = hashlib.sha256((",".join(str(int(t)) for t in ids) + f"|{salt}").encode()).digest()
    return struct.unpack("<Q", h[:8])[0] & ((1 << 63) - 1)


def rows_of(prefix):
    return np.fromfile(prefix + ".rows", dtype=np.int32).reshape(-1, 11).tolist()


def standing(logits):
    """eos rank/p/cum, the top candidate's share, and each nucleus's eos membership."""
    eos_logit = logits[0]
    order = np.argsort(-logits, kind="stable")
    rank = int((logits > eos_logit).sum())
    mx = logits.max()
    w = np.exp(logits - mx)
    z = w.sum()
    p_eos = float(w[0] / z)
    cum = float(np.sort(w)[::-1][: rank + 1].sum() / z)  # mass through eos's rank
    top1 = int(order[0])
    return rank, p_eos, cum, top1, float(w[top1] / z)


def keeps_eos(logits, top_k, top_p):
    rank = int((logits > logits[0]).sum())
    if rank == 0:
        return True
    order = np.argsort(-logits, kind="stable")
    mx = logits[order[0]]
    w = np.exp(logits - mx)
    z = w.sum()
    cum = 0.0
    for i in range(len(order)):
        cum += w[int(order[i])]
        if i == rank:
            return i < top_k and cum / z <= top_p
    return False


def derive(prefix, out):
    lg = np.fromfile(prefix + ".logits", dtype=np.float32).reshape(-1, NCAND)
    fr = np.fromfile(prefix + ".frames", dtype=np.int32).reshape(-1, 11)
    doc = {
        "prompt_rows": len(rows_of(prefix)),
        "frames": len(fr),
        "eos_rank": [], "eos_p": [], "eos_cum": [],
        "top1_cand": [], "top1_p": [],
        "eos_kept_request": [], "eos_kept_ras": [],
        "drawn_main": [],
    }
    for f in range(len(fr)):
        logits = lg[f]
        rank, p_eos, cum, top1, top1_p = standing(logits)
        doc["eos_rank"].append(rank)
        doc["eos_p"].append(round(p_eos, 6))
        doc["eos_cum"].append(round(cum, 6))
        doc["top1_cand"].append(top1)
        doc["top1_p"].append(round(top1_p, 6))
        doc["eos_kept_request"].append(bool(keeps_eos(logits, **REQUEST)))
        doc["eos_kept_ras"].append(bool(keeps_eos(logits, **RAS)))
        doc["drawn_main"].append(int(fr[f][0]))
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"{prefix}: {len(fr)} frames -> {out}")


def analyze(prefix, seed):
    """Replay the pre-patch draws (eos not exempt) and check them against the frames."""
    lg = np.fromfile(prefix + ".logits", dtype=np.float32).reshape(-1, NCAND)
    fr = np.fromfile(prefix + ".frames", dtype=np.int32).reshape(-1, 11)
    window, matched = [-1] * 16, 0
    for f in range(len(fr)):
        logits = lg[f]
        first = draw(logits, 1.0, REQUEST["top_k"], REQUEST["top_p"], seed, f + 1, 0)
        ras = draw(logits, 1.0, RAS["top_k"], RAS["top_p"], seed, f + 1, 1)
        tok = lambda i: EOS if i == 0 else SEM_BEGIN + i - 1
        ft, rt = tok(first), tok(ras)
        repeat = ft in window
        expect = rt if (repeat and ft >= SEM_BEGIN) else ft
        matched += expect == int(fr[f][0])
        window = window[1:] + [int(fr[f][0])]
    print(f"{prefix}: draws match {matched}/{len(fr)} frames (pre-patch rule)")


if __name__ == "__main__":
    if sys.argv[1] == "derive":
        derive(sys.argv[2], sys.argv[3])
    else:
        analyze(sys.argv[2], int(sys.argv[3]))
