"""Opt-in real-checkpoint acceptance for native serial MTP."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BATCH = ROOT.parent / "batch"
sys.path.insert(0, str(ROOT.parent))
from batch.run_native import (
    Stream,
    choose_port,
    require,
    stop_server,
    wait_until_ready,
    write_json,
)
from prefill.run_native import load_cases as load_prefill_cases

COUNTERS = ("mtp_drafted_tokens", "mtp_accepted_tokens", "mtp_rounds")
STEPS = (1, 2, 3, 4)
SEED_COUNTERS = {129: {1: (4, 4, 4), 4: (11, 6, 3)},
                 257: {1: (5, 4, 5), 4: (10, 5, 4)}}


class CompletionStream(Stream):
    def choice(self, ids, finish=None, first=False):
        super().choice(ids, finish, first=first or not self.events)


def load_cases():
    reference = json.loads((ROOT / "reference.json.fixture").read_text())
    cases = {}
    outcomes = set()
    trimmed = False
    for case in reference["cases"]:
        prompt = case["prompt_token_ids"]
        committed = []
        for index, entry in enumerate(case["rounds"]):
            proposals = entry["verified_proposal_ids"]
            matches = entry["accepted_count"]
            winners = entry["greedy_next_ids"]
            rejected = entry["rejected_count"]
            label = f"{case['name']} reference round {index}"
            require(winners[:matches] == proposals[:matches], label)
            require(matches + rejected == len(proposals), label)
            if index:
                require(rejected == 0 or winners[matches] != proposals[matches], label)
                require(entry["target_input_ids"] == [committed[-1], *proposals], label)
                outcomes.add("full" if rejected == 0 else "partial" if matches else "zero")
                require(entry["extra_kind"] == ("correction" if rejected else "bonus"), label)
            else:
                require(entry["target_input_ids"] == prompt and not proposals, label)
                require(entry["extra_kind"] == "ordinary", label)
            position = 0 if index == 0 else len(prompt) + len(committed) - 1
            target_positions = list(range(position, position + len(entry["target_input_ids"])))
            require(entry.get("target_positions", target_positions) == target_positions, label)
            require(entry["worker_sampled_ids"] == winners[:matches + 1], label)
            require(entry["worker_sampled_ids"] == entry["committed_ids"] + entry["trimmed_ids"],
                    label)
            trimmed |= bool(entry["trimmed_ids"])
            if "draft_inputs" in entry:
                first, *tail = entry["draft_inputs"]
                shift = prompt[1:] if index == 0 else []
                require(first["ids"] == shift + entry["committed_ids"], label)
                require(first["sample_row"] == len(first["ids"]) - 1, label)
                require(first["positions"] == list(range(position, position + len(first["ids"]))),
                        label)
                for step, draft_input in enumerate(tail, first["positions"][-1] + 1):
                    require(len(draft_input["ids"]) == 1 and draft_input["sample_row"] == 0 and
                            draft_input["positions"] == [step], label)
            committed.extend(entry["committed_ids"])
            require(entry["processed_after_commit"] == len(prompt) + len(committed) - 1, label)
        require(committed == case["output_ids"] and len(committed) == case["max_tokens"],
                case["name"])
        cases[case["name"]] = prompt
    require(outcomes == {"zero", "partial", "full"} and trimmed, "Reference coverage")
    batch_fixture = BATCH / "fixtures/serial-oracle.json.fixture"
    page = next(case["prompt"] for case in json.loads(batch_fixture.read_text())
                if case["name"] == "page-boundary")
    return cases["capital"], cases["silver-key"], page[:127]


def generate(args, name, case, arm, *, reference=False):
    prompt, budget, stop = case
    executable = args.reference_generate if reference else args.generate
    command = [str(executable), "--checkpoint", str(args.checkpoint),
               "--device", str(args.device), "--prompt-ids", ",".join(map(str, prompt)),
               "--max-new-tokens", str(budget), "--top-logits", "64"]
    if stop is not None:
        command += ["--stop-token", str(stop)]
    if arm != "default":
        command += ["--mtp-steps", str(arm)]
    evidence_name = f"{name}-reference" if reference else name
    evidence = args.evidence / f"{evidence_name}-{arm}"
    evidence.mkdir()
    write_json(evidence / "command.json", command)
    with (
        (evidence / "stdout.json").open("wb") as output,
        (evidence / "stderr.log").open("wb") as errors,
    ):
        subprocess.run(command, check=True, stdout=output, stderr=errors, timeout=300)
    result = json.loads((evidence / "stdout.json").read_text())
    label = f"{name}: steps={arm}"
    require(result["prompt_token_ids"] == prompt, label)
    ids = result["output_token_ids"]
    if stop is not None and stop in ids:
        require(result["finish_reason"] == "stop" and ids.index(stop) == len(ids) - 1 <= budget - 1,
                label)
    else:
        require(result["finish_reason"] == "length" and len(ids) == budget, label)
    require(len(result["step_top_logits"]) == len(ids), label)
    require(all(len(scores) == 64 and scores[0]["token_id"] == token
                for token, scores in zip(ids, result["step_top_logits"], strict=True)), label)
    return result


def compare(target, candidate, steps, name):
    label = f"{name}: steps={steps}"
    for key in ("output_token_ids", "finish_reason", "step_top_logits"):
        require(candidate[key] == target[key], f"{label}: changed {key}")
    if not steps:
        require(all(key not in candidate for key in COUNTERS), f"{label}: draft counters")
        for key in ("model_device_allocation_bytes", "peak_device_allocation_bytes"):
            require(candidate[key] == target[key], f"{label}: changed {key}")
        return
    drafted, accepted, rounds = (candidate[key] for key in COUNTERS)
    output = len(candidate["output_token_ids"])
    if output <= 1:
        require((drafted, accepted, rounds) == (0, 0, 0), f"{label}: unnecessary draft work")
    else:
        require(0 < rounds <= output - 1 and rounds <= drafted <= rounds * steps, label)
        require(0 <= accepted <= min(drafted, output - 1), label)
        require(0 <= (output - 1) - accepted <= rounds, label)
    print(f"{label}: {output} output, {drafted} drafted, {accepted} accepted, {rounds} rounds")


def trim_control(target, budget, stop):
    ids = target["output_token_ids"][:budget]
    stopped = stop is not None and stop in ids
    if stopped:
        ids = ids[:ids.index(stop) + 1]
    return {**target, "output_token_ids": ids, "finish_reason": "stop" if stopped else "length",
            "step_top_logits": target["step_top_logits"][:len(ids)]}


def compare_reference(candidate, reference, steps, name):
    fields = ("output_token_ids", "finish_reason", "step_top_logits")
    for key in (*fields, *(COUNTERS if steps else ())):
        require(candidate[key] == reference[key],
                f"{name}: steps={steps}: changed {key} from the reference generator")
    if not steps:
        require(all(key not in reference for key in COUNTERS), f"{name}: reference draft counters")


def check_seed_carry(args):
    prefill_cases = load_prefill_cases()
    for length, expected_counters in SEED_COUNTERS.items():
        name = f"seed-carry-{length}"
        prompt = prefill_cases[length]["prompt"]
        case = (prompt, 9, None)
        control = generate(args, name, case, 0)
        require(all(key not in control for key in COUNTERS), f"{name}: target-only draft counters")
        if args.reference_generate:
            reference = generate(args, name, case, 0, reference=True)
            compare_reference(control, reference, 0, name)
        for steps, expected in expected_counters.items():
            candidate = generate(args, name, case, steps)
            compare(control, candidate, steps, name)
            counters = tuple(candidate[key] for key in COUNTERS)
            require(counters == expected,
                    f"{name}: steps={steps}: row-wise draft counters {expected} "
                    f"changed to {counters}")
            if args.reference_generate:
                reference = generate(args, name, case, steps, reference=True)
                compare_reference(candidate, reference, steps, name)
    return prompt, control


@contextmanager
def serial_server(args, steps):
    evidence = args.evidence / f"http-{steps}"
    evidence.mkdir()
    port = choose_port()
    command = [str(args.server), "--checkpoint", str(args.checkpoint), "--device",
               str(args.device), "--port", str(port)]
    if steps != "default":
        command += ["--max-active", "1", "--mtp-steps", str(steps)]
    if steps == max(STEPS):
        command.append("--trace-batches")
    write_json(evidence / "command.json", command)
    log_path = evidence / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            wait_until_ready(process, port, log_path)
            yield port, evidence
        finally:
            stop_server(process, None)


def await_serial_retirement(log_path, offset):
    marker = b"batch width=0 decode=0 request_bytes=0\n"
    deadline = time.monotonic() + 5
    with log_path.open("rb") as log:
        while time.monotonic() < deadline:
            log.seek(offset)
            if marker in log:
                return (0, 0, 0)
            time.sleep(0.01)
    raise AssertionError(f"Canceled serial request did not retire. See {log_path}")


def check_recovery(port, evidence, log_path, prompt, target):
    output = target["output_token_ids"]
    cancelled = {"name": "cancel-after-verification", "prompt": prompt,
                 "max_tokens": len(output), "output": output, "finish_reason": "length"}
    offset = log_path.stat().st_size
    with CompletionStream(port, cancelled, budget=128) as stream:
        stream.post()
        stream.open()
        for token in output[:2]:
            stream.choice([token])
        stream.reset()
    retirement = await_serial_retirement(log_path, offset)
    write_json(evidence / "cancel-after-verification.json",
               {**stream.evidence(), "retirement": retirement})

    continuation = {"name": "resume", "prompt": prompt + output[:2],
                    "max_tokens": len(output) - 2, "output": output[2:],
                    "finish_reason": target["finish_reason"]}
    with CompletionStream(port, continuation) as stream:
        stream.post()
        stream.complete()
        write_json(evidence / "resume.json", stream.evidence())


def check_http(args, cases, controls, seed_recovery):
    invalid = [str(args.server), "--max-active", "2", "--mtp-steps", "1"]
    result = subprocess.run(invalid, capture_output=True, text=True, timeout=5)
    require(result.returncode != 0 and "--mtp-steps requires --max-active 1" in result.stderr,
            f"Expected rejection of batched MTP: {result.stderr}")
    for steps in ("default", *STEPS):
        with serial_server(args, steps) as (port, evidence):
            for name, (prompt, budget, stop) in cases.items():
                target = controls[name]
                expected = {"name": name, "prompt": prompt, "max_tokens": budget,
                            "output": target["output_token_ids"],
                            "finish_reason": target["finish_reason"]}
                if stop is not None:
                    expected["stop_token_id"] = stop
                with CompletionStream(port, expected) as stream:
                    stream.post()
                    stream.complete()
                    write_json(evidence / f"{name}.json", stream.evidence())
            if steps == max(STEPS):
                log_path = evidence / "server.log"
                check_recovery(
                    port, evidence, log_path, cases["silver-key"][0], controls["silver-key"])
                seed_evidence = evidence / "seed-carry"
                seed_evidence.mkdir()
                check_recovery(port, seed_evidence, log_path, *seed_recovery)
        print(f"HTTP steps={steps}: exact serial SSE tokens, finish, and usage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=Path, required=True)
    parser.add_argument("--reference-generate", type=Path,
                        help="Previous native generator with row-wise MTP prefill seeds")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/shared/Qwen3.8-27B-FP8"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.generate = args.generate.resolve()
    if args.reference_generate:
        args.reference_generate = args.reference_generate.resolve()
    args.server = args.server.resolve()
    args.evidence.mkdir(parents=True)
    capital, silver, page = load_cases()
    cases = {
        "capital": (capital, 12, None),
        "silver-key": (silver, 24, None),
        "page-127": (page, 16, None),
    }
    controls = {name: generate(args, name, case, "default") for name, case in cases.items()}
    require(all(key not in record for record in controls.values() for key in COUNTERS),
            "Default generation reports no MTP work")
    silver_output = controls["silver-key"]["output_token_ids"]
    edges = {
        "length-one": ("capital", 1, None, 3),
        "length-two": ("capital", 2, None, 4),
        "zero-budget": ("capital", 0, None, 4),
        "prefill-stop": ("capital", 12, 11751, 3),
        "late-stop": ("silver-key", 18, silver_output[6], 3),
        "last-slot-stop": ("silver-key", 7, silver_output[6], 4),
    }
    require(controls["capital"]["output_token_ids"][:3] == [11751, 13, 198],
            "Capital trace prefill and correction")
    for name, case in cases.items():
        compare(controls[name], generate(args, name, case, 0), 0, name)
    for steps in STEPS:
        for name, case in cases.items():
            candidate = generate(args, name, case, steps)
            compare(controls[name], candidate, steps, name)
            if args.reference_generate and name == "silver-key":
                reference = generate(args, name, case, steps, reference=True)
                compare_reference(candidate, reference, steps, name)
    for name, (parent, budget, stop, steps) in edges.items():
        cases[name] = (cases[parent][0], budget, stop)
        controls[name] = trim_control(controls[parent], budget, stop)
        compare(controls[name], generate(args, name, cases[name], steps), steps, name)
    seed_recovery = check_seed_carry(args)
    check_http(args, cases, controls, seed_recovery)


if __name__ == "__main__":
    main()
