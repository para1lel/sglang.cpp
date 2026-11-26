"""Reproducible AOT provisioning of the vendor GDN prefill kernel object.

Compiles the FlashInfer CuTe-DSL Blackwell GDN prefill kernel for the TP2
rank-local geometry (bf16 I/O, fp32 state, 8 query/key heads, 24 value heads,
head dim 128, non-GQA, initial + final state, no checkpoints) directly from
the sources shipped in the pinned flashinfer wheel, into a caller-supplied
fresh output directory. No reliance on any pre-existing JIT cache: the
compilation workspace is a scratch directory created (and removed) here.

The produced object exports the TVM-FFI entry
    __tvm_ffi_gdn_blackwell_prefill_sm_100a_148_bfloat16_float32_8_24_0_1_1_0_0_
    int32_none_none_none_none_1
which gdn_vendor_prefill.cu declares and dispatches through.

Usage (pinned venv python):
    python aot_build_vendor_gdn.py --out /path/to/outdir [--keep-workspace]

Prints the sha256 of the produced object and copies it plus its meta.json to
<out>/gdn_cute_bf16_8_24.o / .meta.json. Run twice into two directories to
verify reproducibility (identical sha256).
"""
import argparse
import glob
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile

EXPECTED = "gdn_blackwell_prefill_sm100a_cute_dsl"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    base = tempfile.mkdtemp(prefix="gdn-aot-", dir=out)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = base

    import torch
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import (
        chunk_gated_delta_rule_sm100,
    )

    if not torch.cuda.is_available():
        print("error: CUDA device required for AOT compilation", file=sys.stderr)
        return 1
    device = "cuda"
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    major, minor, sms = props.major, props.minor, props.multi_processor_count
    if (major, minor, sms) != (10, 0, 148):
        print(
            f"error: this object is a 148-SM sm_100a specialization; board is "
            f"sm_{major}{minor} with {sms} SMs",
            file=sys.stderr,
        )
        return 1

    tokens, hq, hv, dk = 8, 8, 24, 128
    q = torch.randn(tokens, hq, dk, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(tokens, hv, dk, dtype=torch.bfloat16, device=device)
    gate = torch.rand(tokens, hv, dtype=torch.float32, device=device)
    beta = torch.rand(tokens, hv, dtype=torch.float32, device=device)
    o = torch.empty_like(v)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    state = torch.randn(1, hv, dk, dk, dtype=torch.float32, device=device) * 0.02
    out_state = torch.empty_like(state)
    chunk_gated_delta_rule_sm100(
        q, k, v, gate, beta, o, cu, state, out_state, 1.0 / math.sqrt(dk)
    )
    torch.cuda.synchronize()

    objects = []
    for root, _, files in os.walk(base):
        objects.extend(os.path.join(root, name) for name in files if name.endswith(".o"))

    hits = [p for p in objects if EXPECTED in p]
    if not hits:
        print(f"error: no {EXPECTED} object under {base}; found {objects}",
              file=sys.stderr)
        return 1
    obj = hits[0]
    meta = glob.glob(os.path.join(os.path.dirname(obj), "*.json"))

    dest_obj = os.path.join(out, "gdn_cute_bf16_8_24.o")
    shutil.copyfile(obj, dest_obj)
    for m in meta:
        shutil.copyfile(m, os.path.join(out, os.path.basename(m)))

    dest_a = os.path.join(out, "libgdn_cute_bf16_8_24.a")
    subprocess.run(["ar", "rcs", dest_a, dest_obj], check=True)

    if not args.keep_workspace:
        shutil.rmtree(base, ignore_errors=True)

    print(f"object: {dest_obj}")
    print(f"sha256: {sha256(dest_obj)}")
    print(f"source: flashinfer wheel, {EXPECTED}, sm_100a 148 SMs, bf16/fp32 8/24")
    return 0

if __name__ == "__main__":
    sys.exit(main())
