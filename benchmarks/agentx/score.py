"""Score finalized AgentX profiling requests from the pinned harness exports."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

Row = dict[str, Any]


def records(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as stream:
        lines = enumerate(stream, 1)
        number = 0
        try:
            for number, line in lines:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected a JSON object at {path}:{number}")
                yield row
        except ValueError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}") from error


def profiling_rows(path: Path, seen: set[str]) -> Iterator[tuple[str, Row]]:
    for row in records(path):
        metadata = row["metadata"]
        if metadata["benchmark_phase"] != "profiling":
            continue
        request_id = metadata["x_request_id"]
        if not request_id or request_id in seen:
            raise ValueError(f"Missing or duplicate profiling request ID in {path}")
        seen.add(request_id)
        yield request_id, row


def terminal_data(row: Row) -> Iterator[str]:
    for response in reversed(row.get("responses") or []):
        data = "\n".join(
            packet["value"]
            for packet in response.get("packets", [])
            if packet["name"] == "data" and packet.get("value")
        )
        if data:
            yield data


def final_usage(raw: Row, profile: Row | None) -> tuple[str | None, Row | None]:
    metadata = raw["metadata"]
    if metadata.get("context_overflow_skip"):
        return "overflow_skip", None
    if metadata.get("was_cancelled"):
        return "cancelled", None
    status = nonnegative_int(raw.get("status"))
    if (
        raw.get("error")
        or (profile and profile.get("error"))
        or status is None
        or not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES
    ):
        return "request_or_transport_error", None
    return stream_usage(raw)


def stream_usage(raw: Row) -> tuple[str | None, Row | None]:
    events = terminal_data(raw)
    if next(events, None) != "[DONE]":
        return "missing_done", None
    try:
        frame = json.loads(next(events, ""))
        if not isinstance(frame, dict) or frame.get("error"):
            return "missing_finish_choice", None
        usage = {}
        if frame.get("choices") == []:
            usage = frame.get("usage") or {}
            frame = json.loads(next(events, ""))
        choices = (
            frame.get("choices") if isinstance(frame, dict) and not frame.get("error") else None
        )
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return "missing_finish_choice", None
        reason = choices[0].get("finish_reason")
        if not isinstance(reason, str) or not reason or reason in {"error", "abort", "cancelled"}:
            return "missing_finish_choice", None
    except json.JSONDecodeError:
        return "malformed_terminal_event", None
    return None, usage if isinstance(usage, dict) else {}


def nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def trace_blocks(path: Path) -> dict[tuple[str, int], tuple[int, int] | None]:
    manifest = json.loads(path.read_bytes())
    turns = {}
    for conversation in json.loads(manifest["dataset_metadata_json"])["conversations"]:
        for index, turn in enumerate(conversation["turns"]):
            key = conversation["conversation_id"], index
            if key in turns:
                raise ValueError(f"Duplicate manifest turn {key!r}")
            hit = turn.get("theoretical_prefix_cache_hit_blocks")
            total = nonnegative_int(turn.get("theoretical_prefix_cache_total_blocks"))
            turns[key] = (
                (max(0, min(hit, total)), total) if type(hit) is int and total is not None else None
            )
    return turns


def metric(profile: Row, name: str) -> float | None:
    return profile.get("metrics", {}).get(name, {}).get("value")


def collect_usage(
    usage: Row,
    profile: Row | None,
    coverage: Counter[str],
    totals: Counter[str],
    ratios: list[float],
) -> None:
    prompt = nonnegative_int(usage.get("prompt_tokens"))
    completion = nonnegative_int(usage.get("completion_tokens"))
    if prompt is None or completion is None:
        return
    coverage["final_token_usage"] += 1
    totals["input_tokens"] += prompt
    totals["output_tokens"] += completion
    details = usage.get("prompt_tokens_details")
    cached = nonnegative_int(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is not None and cached <= prompt:
        coverage["final_cache_usage"] += 1
        totals["cached_input_tokens"] += cached
    if profile is None:
        return
    if (
        metric(profile, "input_sequence_length") != prompt
        or metric(profile, "output_sequence_length") != completion
    ):
        coverage["profile_token_count_mismatches"] += 1
        return
    if not completion:
        coverage["zero_output_tokens"] += 1
        return
    latency_ms = metric(profile, "request_latency")
    if type(latency_ms) in (float, int) and math.isfinite(latency_ms) and latency_ms > 0:
        ratios.append(latency_ms / 1_000 / completion)
    else:
        coverage["missing_positive_latency"] += 1


def reciprocal_p90(ratios: list[float]) -> float:
    ratios.sort()
    index = (len(ratios) - 1) * 0.90
    lower, upper = math.floor(index), math.ceil(index)
    return 1 / (ratios[lower] + (index - lower) * (ratios[upper] - ratios[lower]))


def harness_coverage(
    aggregate: Row,
    exported_non_skip_rows: int,
    finalized: int,
    trace: tuple[int, int] | None,
) -> tuple[dict[str, float], list[str]]:
    counts = {
        name: aggregate.get(name, {}).get("avg", 0)
        for name in ("request_count", "error_request_count", "skipped_context_overflow_count")
    }
    issues = []
    if exported_non_skip_rows + counts["skipped_context_overflow_count"] != sum(counts.values()):
        issues.append("Profiling raw rows plus overflow skips differ from harness accounting")
    if finalized != counts["request_count"]:
        issues.append("Finalized profiling streams differ from the harness valid request count")
    elif trace and trace[1]:
        theoretical = aggregate.get("theoretical_prefix_cache_hit") or {}
        if (theoretical.get("sum"), theoretical.get("count")) != trace:
            issues.append("Trace blocks differ from the harness theoretical cache aggregate")
    return counts, issues


def request_coverage(
    aggregate: Row,
    unmatched: int,
    coverage: Counter[str],
    chip_seconds: float | None,
) -> list[str]:
    finalized = coverage["finalized"]
    issues = []
    if unmatched:
        issues.append("Some profiling metric rows have no raw SSE row")
    if coverage["profile"] != finalized:
        issues.append("Some finalized streams have no per-request metric row")
    if coverage["profile_token_count_mismatches"] or coverage["missing_positive_latency"]:
        issues.append(
            "Some finalized streams lack matching token counts or positive request latency"
        )
    required = ("final_token_usage", "final_cache_usage", "trace_blocks")
    if not chip_seconds or any(coverage[name] != finalized for name in required):
        issues.append("Some required usage, trace blocks or timestamps are unavailable")
    submission_valid = aggregate.get("metadata", {}).get("submission_valid")
    if aggregate.get("was_cancelled") or submission_valid is False:
        issues.append("The harness marked this run cancelled or an invalid submission")
    return issues


def score(run: Path, manifest: Path, chips: int) -> Row:
    merged = run / "profile_export_raw.jsonl"
    shards = run / "raw_records"
    paths = [merged] if merged.is_file() else sorted(shards.glob("raw_records_*.jsonl"))
    if not paths:
        raise ValueError(f"No raw SSE exports in {run}")
    aggregate = json.loads((run / "profile_export_aiperf.json").read_bytes())
    profiles = dict(profiling_rows(run / "profile_export.jsonl", set()))
    trace = trace_blocks(manifest)
    excluded, coverage, totals = Counter(), Counter(), Counter()
    ratios, raw_ids = [], set()
    first, last = None, None
    for path in paths:
        for request_id, raw in profiling_rows(path, raw_ids):
            metadata = raw["metadata"]
            profile = profiles.pop(request_id, None)
            if profile is not None and profile["metadata"] != metadata:
                raise ValueError(f"Raw and profile metadata differ for request {request_id}")
            problem, usage = final_usage(raw, profile)
            if problem:
                excluded[problem] += 1
                continue
            coverage["finalized"] += 1
            coverage["profile"] += profile is not None
            start = nonnegative_int(metadata.get("request_start_ns"))
            end = nonnegative_int(metadata.get("request_end_ns"))
            if start is not None and end is not None and 0 < start <= end:
                first = start if first is None else min(first, start)
                last = end if last is None else max(last, end)
                coverage["timestamps"] += 1
            blocks = trace.get((metadata.get("conversation_id"), metadata.get("turn_index")))
            if blocks is not None:
                totals["hit_blocks"] += blocks[0]
                totals["total_blocks"] += blocks[1]
                coverage["trace_blocks"] += 1
            collect_usage(usage, profile, coverage, totals, ratios)

    finalized = coverage["finalized"]
    if not finalized:
        raise ValueError(f"No finalized profiling SSE streams in {run}: {dict(excluded)}")
    seconds = (last - first) / 1e9 if coverage["timestamps"] == finalized else None
    chip_seconds = seconds * chips if seconds else None
    trace_total = (totals["hit_blocks"], totals["total_blocks"])
    counts, issues = harness_coverage(
        aggregate,
        len(raw_ids) - excluded["overflow_skip"],
        finalized,
        trace_total if coverage["trace_blocks"] == finalized else None,
    )
    issues.extend(request_coverage(aggregate, len(profiles), coverage, chip_seconds))
    token_seconds = chip_seconds if coverage["final_token_usage"] == finalized else None
    cached = totals["cached_input_tokens"]
    inputs, outputs = totals["input_tokens"], totals["output_tokens"]
    hits, blocks = trace_total
    return {
        "metrics": {
            "p90_e2e_normalized_interactivity_output_tokens_per_s_per_user": (
                reciprocal_p90(ratios) if ratios else None
            ),
            "input_tokens_per_s_per_chip": inputs / token_seconds if token_seconds else None,
            "output_tokens_per_s_per_chip": outputs / token_seconds if token_seconds else None,
            "actual_kv_hit_pct": (
                100 * cached / inputs
                if coverage["final_cache_usage"] == finalized and inputs
                else None
            ),
            "theoretical_kv_hit_pct": (
                100 * hits / blocks if coverage["trace_blocks"] == finalized and blocks else None
            ),
        },
        "observation": {
            "first_start_ns": first,
            "last_response_end_ns": last,
            "seconds": seconds,
            "physical_chips": chips,
            "e2e_sample_count": len(ratios),
            **totals,
        },
        "coverage": {
            "raw_profiling_rows": len(raw_ids),
            "excluded": dict(excluded),
            "finalized": dict(coverage),
            "profiling_metric_rows_without_raw": len(profiles),
            "harness_profiling_counts": counts,
            "issues": issues,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="AgentX artifact directory")
    parser.add_argument("--chips", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="The exact run's dataset mmap manifest"
    )
    args = parser.parse_args()
    try:
        result = score(args.run, args.manifest, args.chips)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
