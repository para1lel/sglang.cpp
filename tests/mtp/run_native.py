"""Opt-in real-checkpoint acceptance for native greedy MTP."""

import argparse
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BATCH = ROOT.parent / "batch"
sys.path.insert(0, str(ROOT.parent))
from batch.run_native import (
    Stream,
    burst,
    choose_port,
    request_body,
    require,
    stop_server,
    wait_until_ready,
    write_json,
)
from prefill.run_native import load_cases as load_prefill_cases

COUNTERS = ("mtp_drafted_tokens", "mtp_accepted_tokens", "mtp_rounds")
STEPS = (1, 2, 3, 4)
BATCH_SPOT = [(2, steps) for steps in STEPS] + [(4, max(STEPS))]
VERIFY_ROWS = re.compile(rb"^batch verify lanes=(\d+) rows=(\d+) ", re.M)
MTP_ROUND = re.compile(rb"drafted=(\d+) accepted=(\d+) verified=(\d+) replayed=\d+ boundary=(\d+)")
SEED_LENGTHS = (129, 257)
SEED_STEPS = (1, max(STEPS))


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
    check_draft_totals(candidate, steps, label)


def check_draft_totals(result, steps, label, *, terminal_at_budget=False):
    drafted, accepted, rounds = (result[key] for key in COUNTERS)
    output = len(result["output_token_ids"])
    decode_outputs = max(output - 1, 0)
    require(0 <= rounds <= decode_outputs, label)
    require(rounds <= drafted <= rounds * steps, label)
    require(0 <= accepted <= min(drafted, decode_outputs), label)
    # One terminal proposal can suppress the correction or bonus. One final
    # output can use an ordinary target step without a draft round.
    terminal_proposal = accepted > 0 and (terminal_at_budget or result["finish_reason"] == "stop")
    require(rounds - terminal_proposal <= decode_outputs - accepted <= rounds + 1, label)
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
    for key in fields:
        require(candidate[key] == reference[key],
                f"{name}: steps={steps}: changed {key} from the reference generator")
    if steps:
        label = f"{name}: reference steps={steps}"
        check_draft_totals(reference, steps, label, terminal_at_budget=True)
        require(reference["mtp_accepted_tokens"] > 0, f"{label}: no accepted proposals")
    else:
        require(all(key not in reference for key in COUNTERS), f"{name}: reference draft counters")


def check_seed_carry(args):
    prefill_cases = load_prefill_cases()
    seeds = {}
    for length in SEED_LENGTHS:
        name = f"seed-carry-{length}"
        prompt = prefill_cases[length]["prompt"]
        case = (prompt, 9, None)
        control = generate(args, name, case, 0)
        require(all(key not in control for key in COUNTERS), f"{name}: target-only draft counters")
        if args.reference_generate:
            reference = generate(args, name, case, 0, reference=True)
            compare_reference(control, reference, 0, name)
        candidates = {}
        for steps in SEED_STEPS:
            candidate = generate(args, name, case, steps)
            compare(control, candidate, steps, name)
            require(candidate["mtp_accepted_tokens"] > 0, f"{name}: no accepted proposals")
            candidates[steps] = candidate
            if args.reference_generate:
                reference = generate(args, name, case, steps, reference=True)
                compare_reference(candidate, reference, steps, name)
        seeds[name] = (prompt, control, candidates)
    return seeds


@contextmanager
def mtp_server(args, steps, width=1, *, token_budget=None):
    suffix = f"{steps}" if width == 1 else f"w{width}-k{steps}"
    if token_budget is not None:
        suffix += f"-rows{token_budget}"
    evidence = args.evidence / f"http-{suffix}"
    evidence.mkdir()
    port = choose_port()
    command = [str(args.server), "--checkpoint", str(args.checkpoint), "--device",
               str(args.device), "--port", str(port)]
    if steps != "default":
        command += ["--max-active", str(width), "--mtp-steps", str(steps)]
    if token_budget is not None:
        command.extend(("--max-num-batched-tokens", str(token_budget)))
    if width > 1 or steps in SEED_STEPS or token_budget is not None:
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


def await_retirement(log_path, offset):
    marker = b"batch width=0 decode=0 request_bytes=0\n"
    deadline = time.monotonic() + 5
    with log_path.open("rb") as log:
        while time.monotonic() < deadline:
            log.seek(offset)
            if marker in log:
                return (0, 0, 0)
            time.sleep(0.01)
    raise AssertionError(f"Canceled request did not retire. See {log_path}")


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
    retirement = await_retirement(log_path, offset)
    write_json(evidence / "cancel-after-verification.json",
               {**stream.evidence(), "retirement": retirement})

    continuation = {"name": "resume", "prompt": prompt + output[:2],
                    "max_tokens": len(output) - 2, "output": output[2:],
                    "finish_reason": target["finish_reason"]}
    with CompletionStream(port, continuation) as stream:
        stream.post()
        stream.complete()
        write_json(evidence / "resume.json", stream.evidence())


def check_cli_policy(args):
    cases = {}
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for name, binary, flags in (("generate", args.generate, ["--prompt-ids", "17"]),
                                ("server", args.server, ["--max-active", "2"])):
        command = [str(binary), "--help"]
        help_result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                     timeout=5)
        help_lines = help_result.stdout.splitlines()
        require(help_result.returncode == 0 and not help_result.stderr and
                any(re.search(r"(?:default 4\b|\b4 by default)", line)
                    for line in help_lines if "--mtp-steps" in line or "MTP steps" in line),
                f"{name}: help must advertise K4 as the default")
        require("--mtp-k" in help_result.stdout, f"{name}: help must advertise the depth alias")
        if name == "server":
            require(any("0 steps" in line and "logprobs" in line for line in help_lines),
                    "server: help must describe the K0 fallback for logprobs")
        cases[f"{name}-help"] = {"command": command, "stdout": help_result.stdout}
        sampling = (
            (("seed", ["--seed", "9223372036854775808"], "Invalid number for --seed"),
             ("temperature", ["--temperature", "-1"], "Expected finite --temperature >= 0"))
            if name == "generate" else
            (("seed", ["--seed", "1"], "Invalid argument --seed"),
             ("temperature", ["--temperature", "1"], "Invalid argument --temperature"))
        )
        validations = (
            ("depth", ["--mtp-steps", "5"], "Expected --mtp-steps in [0, 4]"),
            ("depth-alias", ["--mtp-k", "5"], "Expected --mtp-steps in [0, 4]"),
            *sampling,
        )
        for label, options, diagnostic in validations:
            command = [str(binary), *flags, *options]
            result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                    timeout=5)
            require(result.returncode != 0 and not result.stdout and diagnostic in result.stderr,
                    f"{name} {label}: {result.stderr}")
            cases[f"{name}-{label}"] = {
                "command": command, "status": result.returncode, "stderr": result.stderr,
            }
    write_json(args.evidence / "cli-policy.json", cases)


def check_rng_policy(port, evidence):
    completions = "/v1/completions"
    chat = "/v1/chat/completions"
    model = "Qwen3.8-27B-FP8"
    ids = request_body({"prompt": [17], "max_tokens": 0})
    text = {"model": model, "prompt": "Spain", "max_tokens": 0}
    messages = {"model": model, "messages": [{"role": "user", "content": "Spain"}],
                "max_tokens": 0, "temperature": 0}
    sampled = {"temperature": 1, "top_p": 0.9, "top_k": 32, "seed": -1, "max_tokens": 3}
    cases = (
        ("ids-greedy", completions, ids, None),
        ("ids-temperature", completions, ids | {"temperature": 1},
         "Expected the frozen benchmark streaming options"),
        ("ids-seed", completions, ids | {"seed": 1},
         "Expected the frozen benchmark streaming options"),
        ("ids-top-p", completions, ids | {"top_p": 0.5},
         "Expected the frozen benchmark streaming options"),
        ("text-greedy", completions, text, None),
        ("text-seed", completions, text | {"seed": 0}, None),
        ("text-sampled", completions, text | sampled, None),
        ("chat-greedy", chat, messages, None),
        ("chat-sampled", chat, messages | sampled, None),
        ("chat-seed", chat, messages | {"seed": 0}, None),
        ("chat-top-p", chat, messages | {"top_p": 0.5}, None),
    )
    results = {}
    for name, route, body, diagnostic in cases:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.request("POST", route, json.dumps(body),
                               {"Content-Type": "application/json", "Connection": "close"})
            with connection.getresponse() as response:
                received = response.read().decode("utf-8")
                status = response.status
                expect = 400 if diagnostic else 200
                require(status == expect, f"{name}: HTTP {status}: {received}")
                if diagnostic:
                    require(response.headers.get_content_type() == "application/json" and
                            json.loads(received)["error"]["message"] == diagnostic,
                            f"{name}: {received}")
                elif name == "ids-greedy":
                    require(response.headers.get_content_type() == "text/event-stream" and
                            received.endswith("data: [DONE]\n\n"), f"{name}: {received}")
                else:
                    require(json.loads(received)["choices"][0]["finish_reason"] == "length",
                            f"{name}: {received}")
                results[name] = {
                    "route": route, "request": body, "status": status, "body": received,
                }
        finally:
            connection.close()
    write_json(evidence / "rng-policy.json", results)


def check_seed_http(port, evidence, log_path, name, seed, steps):
    prompt, target, candidates = seed
    output = target["output_token_ids"]
    case = {"name": name, "prompt": prompt, "max_tokens": len(output),
            "finish_reason": target["finish_reason"], "output": output}
    offset = log_path.stat().st_size
    with CompletionStream(port, case) as stream:
        stream.post()
        stream.complete()
    rows = read_rounds(log_path, offset, output, steps)
    first_drafted, first_accepted, _, _ = rows[0]
    require(first_drafted == steps and first_accepted > 0,
            f"{name}: initial verifier must accept a proposal from the prefill seed: {rows}")
    require(draft_totals(rows) == tuple(candidates[steps][key] for key in COUNTERS),
            f"{name}: CLI and HTTP speculative totals differ: {rows}")
    write_json(evidence / f"{name}.json", {**stream.evidence(), "mtp_rounds": rows})


def check_http(args, cases, controls, seeds):
    for steps in ("default", *STEPS):
        with mtp_server(args, steps) as (port, evidence):
            if steps in ("default", max(STEPS)):
                check_rng_policy(port, evidence)
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
            log_path = evidence / "server.log"
            if steps in SEED_STEPS:
                for name, seed in seeds.items():
                    check_seed_http(port, evidence, log_path, name, seed, steps)
            if steps == max(STEPS):
                check_recovery(
                    port, evidence, log_path, cases["silver-key"][0], controls["silver-key"])
                seed_name = f"seed-carry-{max(SEED_LENGTHS)}"
                seed_evidence = evidence / seed_name
                seed_evidence.mkdir()
                prompt, control, _ = seeds[seed_name]
                check_recovery(port, seed_evidence, log_path, prompt, control)
        print(f"HTTP steps={steps}: exact single-request SSE tokens, finish, and usage")


def check_batch_http(args, prompt, target, settings):
    tokens = target["output_token_ids"]
    for width, steps in settings:
        with mtp_server(args, steps, width) as (port, evidence):
            cases = [
                {"name": f"lane-{lane}", "prompt": prompt, "max_tokens": len(tokens),
                 "finish_reason": target["finish_reason"], "output": tokens}
                for lane in range(width)
            ]
            log_path = evidence / "server.log"
            with log_path.open("rb") as log:
                log.seek(0, 2)
                burst(port, log_path, evidence / "burst.json", cases, full_decode=True)
                rows = [tuple(map(int, group)) for group in VERIFY_ROWS.findall(log.read())]
            if width == 4 and steps == max(STEPS):
                check_rng_policy(port, evidence)
        require((width, width * (steps + 1)) in rows,
                f"Missing joint verifier at width {width}, steps {steps}: {log_path}")
        print(f"HTTP width={width} steps={steps}: exact SSE and joint target verification")


def check_round_accounting(args, prompt, target, candidate, token_budget):
    output = target["output_token_ids"]
    case = {"name": "round-accounting", "prompt": prompt, "max_tokens": len(output),
            "finish_reason": "length", "output": output}
    with mtp_server(args, 1, token_budget=token_budget) as (port, evidence):
        with CompletionStream(port, case) as stream:
            stream.post()
            stream.complete()
            write_json(evidence / "round-accounting.json", stream.evidence())
    rows = read_rounds(evidence / "server.log", 0, output, 1)
    drafted, accepted, positive = draft_totals(rows)
    zero_proposal = sum(drafted == 0 and verified == 1 for drafted, _, verified, _ in rows)
    require(rows[-1] == (0, 0, 0, 1), rows)
    if token_budget == 1:
        require(drafted == accepted == positive == 0 and zero_proposal > 0, rows)
    else:
        require((drafted, accepted, positive) == tuple(candidate[key] for key in COUNTERS)
                and positive > 0 and zero_proposal == 0, rows)


def read_rounds(log_path, offset, output, steps):
    with log_path.open("rb") as log:
        log.seek(offset)
        rows = [tuple(map(int, row)) for row in MTP_ROUND.findall(log.read())]
    require(rows and all(0 <= accepted <= drafted <= steps and
                         verified == (0 if boundary else drafted + 1) and
                         (not boundary or (drafted, accepted, verified, boundary) == (0, 0, 0, 1))
                         for drafted, accepted, verified, boundary in rows), rows)
    _, accepted, positive = draft_totals(rows)
    zero_proposal = sum(drafted == 0 and verified == 1 for drafted, _, verified, _ in rows)
    ordinary_tail = sum(boundary for _, _, _, boundary in rows)
    # For these requests that reach the output limit, each verification round
    # emits one target token in addition to its accepted proposals.
    require(len(output) == 1 + accepted + positive + zero_proposal + ordinary_tail, rows)
    require(ordinary_tail <= 1 and (not ordinary_tail or rows[-1] == (0, 0, 0, 1)), rows)
    return rows


def draft_totals(rows):
    return (sum(drafted for drafted, _, _, _ in rows),
            sum(accepted for _, accepted, _, _ in rows),
            sum(drafted > 0 for drafted, _, _, _ in rows))


def check_batch_spot(args, page):
    name = "page-127"
    case = (page, 10, None)
    target = generate(args, name, case, 0)
    require(all(key not in target for key in COUNTERS), f"{name}: target-only draft counters")
    for steps in STEPS:
        candidate = generate(args, name, case, steps)
        compare(target, candidate, steps, name)
        if steps == 1:
            for token_budget in (32768, 1):
                check_round_accounting(args, page, target, candidate, token_budget)
    check_batch_http(args, page, target, BATCH_SPOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=Path, required=True)
    parser.add_argument("--reference-generate", type=Path,
                        help="Previous native generator with row-wise MTP prefill seeds")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data0/shared/Qwen3.8-27B-FP8"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--batch-spot-check", action="store_true",
                        help="Check generation at depths 0–4 and five joint HTTP batch shapes")
    args = parser.parse_args()
    args.generate = args.generate.resolve()
    if args.reference_generate:
        args.reference_generate = args.reference_generate.resolve()
    args.server = args.server.resolve()
    args.evidence.mkdir(parents=True)
    check_cli_policy(args)
    capital, silver, page = load_cases()
    if args.batch_spot_check:
        check_batch_spot(args, page)
        return
    cases = {
        "capital": (capital, 12, None),
        "silver-key": (silver, 24, None),
        "page-127": (page, 16, None),
    }
    controls = {name: generate(args, name, case, 0) for name, case in cases.items()}
    require(all(key not in record for record in controls.values() for key in COUNTERS),
            "Explicit K0 generation reports no MTP work")
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
    default_candidate = generate(args, "silver-key", cases["silver-key"], "default")
    compare(controls["silver-key"], default_candidate, max(STEPS), "silver-key default")
    for steps in STEPS:
        for name, case in cases.items():
            candidate = generate(args, name, case, steps)
            compare(controls[name], candidate, steps, name)
            if name == "silver-key" and steps == max(STEPS):
                require(all(default_candidate[key] == candidate[key] for key in COUNTERS)
                        and default_candidate["mtp_rounds"] > 0,
                        "Default generation matches explicit K4 speculative work")
            if args.reference_generate and name == "silver-key":
                reference = generate(args, name, case, steps, reference=True)
                compare_reference(candidate, reference, steps, name)
    for name, (parent, budget, stop, steps) in edges.items():
        cases[name] = (cases[parent][0], budget, stop)
        controls[name] = trim_control(controls[parent], budget, stop)
        compare(controls[name], generate(args, name, cases[name], steps), steps, name)
    seeds = check_seed_carry(args)
    check_http(args, cases, controls, seeds)
    check_batch_http(args, page, controls["page-127"],
                     [(width, steps) for width in (2, 3, 4) for steps in STEPS])


if __name__ == "__main__":
    main()
