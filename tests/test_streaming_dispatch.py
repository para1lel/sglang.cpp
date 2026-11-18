import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECK = Path(__file__).resolve().parents[1] / "benchmarks" / "streaming" / "check_vllm.py"
PREFILL = (
    "void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, "
    "flashinfer::KernelTraits<(flashinfer::MaskMode)1, 64u, 1u, 4u, 16u, 16u, 4u, 1u, "
    "(flashinfer::PosEncodingMode)0, __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16, float, int, "
    "flashinfer::DefaultAttention<false, false, false, false>, true, false, false>, "
    "PagedParams>(PagedParams)"
)
DECODE = (
    "fmhaSm100aKernel_QBfloat16KvE4m3OBfloat16H256PagedKvCausalP64MultiCtasKv"
    "CgaVarSeqSeparateTransKVQ8Kv128StaticGroupedSwapsAbForGen"
)
GDN_PREFILL = (
    "kernel_cutlass_kernel_flashinfergdn_kernelsblackwellgated_delta_net_chunked"
    "GatedDeltaNetChunkedKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_"
    "PermutationMNK____MMAAtom_ThrID10_ShapeMNK6_0"
)
GDN_DECODE = "fused_recurrent_gated_delta_rule_packed_decode_kernel"


def probe_events():
    events = [
        {"cat": "kernel", "ph": "X", "name": name}
        for name, count in ((PREFILL, 16), (DECODE, 32), (GDN_PREFILL, 48), (GDN_DECODE, 96))
        for _ in range(count)
    ]
    return [*events, {"cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch"}]


def check_trace(folder, name):
    return subprocess.run(
        [sys.executable, "-IB", str(CHECK), "trace", str(folder), name],
        text=True, capture_output=True, timeout=10,
    )


class StreamingDispatchTest(unittest.TestCase):
    def test_requires_single_trace_for_requested_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            text = json.dumps({"traceEvents": probe_events()})
            plain = folder / "W1_rank0.pt.trace.json"
            plain.write_text(text, encoding="utf-8")
            accepted = check_trace(folder, "W1")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            summary = json.loads(accepted.stdout)
            self.assertEqual(summary["trace_files"], [plain.name])
            self.assertEqual(summary["graph_launches"], {"cudaGraphLaunch": 1})

            missing = check_trace(folder, "W2")
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("W2 requires exactly one profiler trace, found 0", missing.stderr)

            (folder / "W1_rank0.second.pt.trace.json.gz").write_bytes(gzip.compress(text.encode()))
            duplicate = check_trace(folder, "W1")
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("found 2", duplicate.stderr)

    def test_rejects_wrong_routes_and_unexecuted_graph_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            trace = folder / "W2_rank0.pt.trace.json.gz"
            events = probe_events()
            trace.write_bytes(gzip.compress(json.dumps({"traceEvents": events}).encode()))
            accepted = check_trace(folder, "W2")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            for name, alternative, expected_error in (
                (PREFILL, PREFILL.replace("__nv_bfloat16", "__half", 1), "FA2 prefill"),
                (DECODE, "flashinfer::BatchDecodeWithPagedKVCacheKernel", "TRTLLM-Gen decode"),
                (GDN_PREFILL, "_causal_conv1d_fwd_kernel", "chunked GDN prefill"),
                (GDN_DECODE, "_causal_conv1d_update_kernel", "recurrent GDN decode"),
            ):
                with self.subTest(route=expected_error):
                    wrong = [
                        {**event, "name": alternative} if event["name"] == name else event
                        for event in events
                    ]
                    trace.write_bytes(gzip.compress(json.dumps({"traceEvents": wrong}).encode()))
                    rejected = check_trace(folder, "W2")
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn(expected_error, rejected.stderr)

            for graph_events, expected_error in (
                ([], "FA2 prefill"),
                (events[:-1], "requires CUDA graph launches"),
                ([*events[:-1], {"cat": "user_annotation", "ph": "X",
                                "name": "cudaGraphLaunch"}], "requires CUDA graph launches"),
                ([*events[:-1], {**events[-1], "ph": "i"}], "requires CUDA graph launches"),
            ):
                with self.subTest(error=expected_error, events=len(graph_events)):
                    trace.write_bytes(
                        gzip.compress(json.dumps({"traceEvents": graph_events}).encode())
                    )
                    rejected = check_trace(folder, "W2")
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn(expected_error, rejected.stderr)


if __name__ == "__main__":
    unittest.main()
