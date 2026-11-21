"""Check paired-control receipts offline using the fixed W1 and W2 requests."""

import json
import shlex
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "paired"))
import identity


GPU = "GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c"
OTHER_GPU = "GPU-3c08cbc3-d8b3-c723-c086-a775f6a17f2a"
PLACEMENT = ["numactl", "--localalloc", "taskset", "--cpu-list"]
GROUP = "W1-W2"


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_command(run, role, cpus, args):
    (run / f"{role}-command.sh").write_text(shlex.join([*PLACEMENT, cpus, *args]))


def write_receipts(capture, index, uuid):
    (capture / "gpu.txt").write_text(f"{index} {uuid}\n")
    (capture / "cpus.txt").write_text("server=24-33\nclient=34-35\nmemory=localalloc\n")
    (capture / "devices.csv").write_text(
        f"index, uuid, name, driver_version\n{index}, {uuid}, NVIDIA B200, 590.48.01\n")


def write_compute(run, uuid, pid):
    for sample in ("ready", "measurement-end"):
        (run / f"compute-{sample}.csv").write_text(f"{uuid}, {pid}, worker, 80000 MiB\n")


def stream_record(case, request_hash, name, phase, cohort):
    tokens = list(range(case["output_tokens"]))
    prompt = case["prompt_token_ids"]
    usage = {"prompt_tokens": len(prompt), "completion_tokens": len(tokens),
             "prompt_tokens_details": {"cached_tokens": 0}}
    payloads = [
        {"choices": [{"index": 0, "token_ids": tokens[:1], "prompt_token_ids": prompt}]},
        {"choices": [{"index": 0, "token_ids": tokens[1:], "finish_reason": "length"}]},
        {"choices": [], "usage": usage},
    ]
    data = [json.dumps(payload) for payload in payloads] + ["[DONE]"]
    events = [{"lines": ["data: " + payload], "ns": stamp, "read": index}
              for index, (payload, stamp) in enumerate(zip(data, (20, 50, 60, 70)), start=1)]
    duration = 70
    aggregate_rate = (len(tokens) + len(prompt)) * 1e9 / duration
    return {"workload": name, "phase": phase, "cohort": cohort, "http_status": 200,
            "submit_ns": 0, "headers_ns": 10, "request_sha256": request_hash,
            "events": events, "output_ids": tokens,
            "metrics": {"ttft_ms": 20 / 1e6, "tpot_ms": 30 / (len(tokens) - 1) / 1e6,
                        "e2e_ms": duration / 1e6,
                        "output_tokens_per_second": len(tokens) * 1e9 / duration,
                        "requests_per_second": 1e9 / duration,
                        "aggregate_tokens_per_second": aggregate_rate}}


def write_measurement(run, hashes, freeze_baseline=True):
    measurement = run / "measurement"
    measurement.mkdir()
    manifest = identity.stream_identity(ROOT, hashes)
    cases = identity.streaming.load_cases(ROOT / "benchmarks" / "streaming")
    records = {name: [] for name in cases}
    for name, (case, _) in cases.items():
        request = identity.streaming.request_body(case)
        (measurement / f"{name}.request.json").write_bytes(identity.streaming.json_bytes(request))
    with (measurement / "events.jsonl").open("w") as events:
        for phase, count in manifest["schedule"]:
            for cohort in range(1, count + 1):
                for name, (case, _) in cases.items():
                    record = stream_record(case, manifest["request_sha256"][name],
                                           name, phase, cohort)
                    events.write(json.dumps(record) + "\n")
                    records[name].append(record)
    write_json(measurement / "summary.json", {"manifest": manifest, "workloads": {
        name: identity.streaming.summarize(items, freeze_baseline)
        for name, items in records.items()}})
    (run / "client-exit-code.txt").write_text("0\n")


def write_saved_worker(run, checkpoint, reference):
    worker = reference["observed_vllm_settings"]
    worker.update(
        scheduler={"max_model_len": 2560, "max_num_batched_tokens": 32768, "max_num_seqs": 32,
                   "max_num_scheduled_tokens_effective": 32768, "enable_chunked_prefill": True},
        parallel={"tp": 1, "dp": 1, "pp": 1}, quantization="fp8", skip_tokenizer_init=True,
        flashinfer="0.7.4",
        kv_transfer=False, kv_offloading_size=None, cpu_offload="cpu_offload_gb=0.0, "
        "offload_group_size=0, ")
    worker["cache"]["kv_cache_memory_bytes"] = 10 << 30
    write_json(run / "effective-worker.json", {"results": [worker]})
    write_command(run, "server", "24-33", [
        "python", "-m", "vllm.entrypoints.openai.api_server", "--model", str(checkpoint),
        "--max-model-len", "2560", "--max-num-seqs", "32", "--max-num-batched-tokens", "32768",
        "--generation-config", "vllm", "--seed", "0", "--stream-interval", "1",
        "--language-model-only", "--no-enable-prefix-caching"])
    (run / "server-pid.txt").write_text("1000\n")
    (run / "server.log").write_text(
        "(APIServer pid=1000) Starting engine\n"
        f"(EngineCore pid=1001) Initializing a V1 LLM engine with config: model='{checkpoint}'\n")
    write_compute(run, GPU, 1001)


class PairedIdentityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        saved = self.root / "saved"
        output = self.root / "native"
        capture = saved / f"vllm-{GROUP}"
        self.saved_run = capture / GROUP
        self.saved_run.mkdir(parents=True)
        output.mkdir()
        self.reference = identity.concurrent.reference()
        checkpoint = self.root / self.reference["model"]
        checkpoint.mkdir()
        weight = checkpoint / "weights.safetensors"
        weight.write_bytes(b"fixed checkpoint fixture")
        manifest = saved / "checkpoint.sha256"
        manifest.write_text(f"{identity.digest(weight)}  {weight.name}\n")
        self.args = SimpleNamespace(
            mode="matched", saved=saved, output=output, protocol=ROOT, groups=[GROUP], group=GROUP,
            checkpoint=checkpoint, checkpoint_manifest=manifest, warmups=2, measured=5,
            fixtures=self.root / "fixtures", allow_count_mismatch=False)
        self.reference["checkpoint_manifest_sha256"] = identity.digest(manifest)
        patcher = patch.object(identity.concurrent, "reference", side_effect=self.fixture_reference)
        patcher.start()
        self.addCleanup(patcher.stop)
        write_receipts(capture, 2, GPU)
        write_receipts(output, 0, GPU)
        (capture / "effective-environment.txt").write_text(f"CUDA_VISIBLE_DEVICES={GPU}\n")
        hashes = identity.current_protocol(GROUP, ROOT)
        (capture / "protocol.sha256").write_text("".join(
            f"{sha}  /saved/benchmarks/{name}\n" for name, sha in hashes.items()))
        write_saved_worker(self.saved_run, checkpoint, self.fixture_reference())
        write_command(self.saved_run, "client", "34-35", [
            "python", str(ROOT / "benchmarks/streaming/client.py"), "--freeze-baseline"])
        write_measurement(self.saved_run, hashes)

    def fixture_reference(self):
        return deepcopy(self.reference)

    def test_same_uuid_can_have_different_logical_index(self):
        plan = identity.plan(self.args)
        self.assertEqual(plan["groups"][GROUP]["planned_comparison"], "same-device matched-cap")
        self.assertEqual(plan["groups"][GROUP]["vllm"]["device"]["uuid"], GPU.lower())

    def test_equal_counts_cannot_hide_device_mismatch(self):
        self.args.allow_count_mismatch = True
        write_receipts(self.args.output, 2, OTHER_GPU)
        with self.assertRaisesRegex(ValueError, "saved device"):
            identity.plan(self.args)

    def test_compare_process_uuid_at_measurement_end(self):
        (self.saved_run / "compute-measurement-end.csv").write_text(f"{OTHER_GPU}, 1001, worker\n")
        with self.assertRaisesRegex(ValueError, "compute PID 1001"):
            identity.plan(self.args)

    def test_compare_saved_request_body(self):
        with (self.saved_run / "measurement" / "W1.request.json").open("ab") as request:
            request.write(b" ")
        with self.assertRaisesRegex(ValueError, "saved streaming request digest"):
            identity.plan(self.args)

    def test_compare_observed_scheduler_cap(self):
        receipt = self.saved_run / "effective-worker.json"
        worker = identity.load(receipt)
        worker["results"][0]["scheduler"]["max_num_scheduled_tokens_effective"] = 8192
        write_json(receipt, worker)
        with self.assertRaisesRegex(ValueError, "effective vLLM scheduler"):
            identity.plan(self.args)

    def test_reject_vllm_eager_command_with_graph_receipt(self):
        command = self.saved_run / "server-command.sh"
        command.write_text(command.read_text() + " --enforce-eager")
        with self.assertRaisesRegex(ValueError, "effective vLLM scheduler"):
            identity.plan(self.args)

    def test_compare_protocol_digest(self):
        receipt = self.saved_run.parent / "protocol.sha256"
        contents = receipt.read_text().splitlines()
        receipt.write_text("0" * 64 + contents[0][64:] + "\n" + "\n".join(contents[1:]) + "\n")
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            identity.plan(self.args)

    def test_replay_detects_summary_metric_changes_with_equal_counts(self):
        receipt = self.saved_run / "measurement" / "summary.json"
        report = identity.load(receipt)
        report["workloads"]["W1"]["ttft_ms"]["p95"] = 1234
        write_json(receipt, report)
        with self.assertRaisesRegex(ValueError, "streaming summary differs from replay"):
            identity.plan(self.args)

    def test_check_actual_checkpoint_bytes(self):
        (self.args.checkpoint / "weights.safetensors").write_bytes(b"different checkpoint fixture")
        with self.assertRaises(identity.subprocess.CalledProcessError):
            identity.plan(self.args)

    def test_published_comparison_has_cross_cap_and_device_labels(self):
        self.args.mode = "frozen"
        self.args.saved = None
        self.args.groups = ["W5"]
        with patch.object(identity, "load_lanes") as fixtures:
            first = identity.plan(self.args)["groups"]["W5"]
            write_receipts(self.args.output, 2, OTHER_GPU)
            second = identity.plan(self.args)["groups"]["W5"]
        self.assertEqual(fixtures.call_count, 2)
        self.assertEqual(first["planned_comparison"], "published frozen reference (cross-cap)")
        self.assertEqual(second["planned_comparison"],
                         "published frozen reference (cross-device, cross-cap)")

    def prepare_native(self):
        output = self.args.output
        plan_path = output / "comparison-plan.json"
        write_json(plan_path, identity.plan(self.args))
        self.args.plan_sha256 = identity.digest(plan_path)
        run = output / GROUP
        run.mkdir()
        (run / "server-pid.txt").write_text("2000\n")
        write_compute(run, GPU, 2000)
        write_command(run, "server", "24-33", [
            "sglang-server", "--checkpoint", str(self.args.checkpoint), "--device", "0",
            "--max-active", "32", "--max-num-batched-tokens", "32768",
            "--fp8-prefill", "tensor-core", "--prefix-cache-entries", "0", "--mtp-steps", "0",
            "--batch-cuda-graphs"])
        write_command(run, "client", "34-35", [
            "python", str(ROOT / "benchmarks/streaming/client.py")])
        (output / "native-environment.txt").write_text(f"CUDA_VISIBLE_DEVICES={GPU}\n")
        write_json(run / "health.json", {"max_active": 32, "max_num_batched_tokens": 32768})
        hashes = identity.current_protocol(GROUP, ROOT)
        (output / "protocol.sha256").write_text("".join(
            f"{sha}  /native/benchmarks/{name}\n" for name, sha in hashes.items()))
        write_measurement(run, hashes, freeze_baseline=False)
        return run

    def test_publish_only_after_native_process_is_observed(self):
        run = self.prepare_native()
        write_compute(run, OTHER_GPU, 2000)
        with self.assertRaisesRegex(ValueError, "compute PID 2000"):
            identity.finish(self.args)
        write_compute(run, GPU, 2000)
        result = identity.finish(self.args)
        self.assertEqual(result["comparison"], "same-device matched-cap")
        self.assertFalse(result["quality"]["cross_engine_ids_compared"])

    def test_reject_relabelled_comparison_plan(self):
        self.prepare_native()
        path = self.args.output / "comparison-plan.json"
        plan = identity.load(path)
        plan["groups"][GROUP]["planned_comparison"] = "published frozen reference (cross-cap)"
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            identity.finish(self.args)
        self.args.plan_sha256 = identity.digest(path)
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            identity.finish(self.args)

    def test_reject_finish_in_another_mode(self):
        self.prepare_native()
        self.args.mode = "frozen"
        with self.assertRaisesRegex(ValueError, "plan mode differs"):
            identity.finish(self.args)

    def test_reject_wrong_native_client(self):
        run = self.prepare_native()
        write_command(run, "client", "34-35", ["python", "client.py"])
        with self.assertRaisesRegex(ValueError, "unexpected streaming client"):
            identity.finish(self.args)

    def test_reject_native_mtp_alias(self):
        run = self.prepare_native()
        command = run / "server-command.sh"
        command.write_text(command.read_text() + " --mtp-k 1")
        with self.assertRaisesRegex(ValueError, "native command differs"):
            identity.finish(self.args)

    def test_replay_detects_native_summary_change(self):
        run = self.prepare_native()
        receipt = run / "measurement" / "summary.json"
        report = identity.load(receipt)
        report["workloads"]["W1"]["ttft_ms"]["p95"] = 1234
        write_json(receipt, report)
        with self.assertRaisesRegex(ValueError, "streaming summary differs from replay"):
            identity.finish(self.args)
