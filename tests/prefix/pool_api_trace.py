"""Check startup and serving GPU allocation lifetimes in an Nsight SQLite export."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator


class ApiCall(TypedDict):
    start: int
    end: int
    thread: int
    correlation: int
    name: str
    status: int | None
    kind: str


class MemoryEvent(TypedDict):
    phase: str
    role: str
    device: int
    address: int
    bytes: int
    kind: str
    operation: str
    api: str | None
    ownership: str


class ApiCount(TypedDict):
    phase: str
    role: str
    kind: str
    api: str
    status: int | None
    count: int


class Inventory(TypedDict):
    markers_ns: dict[str, int]
    api_counts: list[ApiCount]
    memory_events: list[MemoryEvent]


MARKER = "prefix-pool/"
ALLOC = re.compile(
    r"^(?:cuda(?:Malloc(?:3D|Async|Array|FromPoolAsync|Managed|MipmappedArray|Pitch)*|"
    r"Free(?:Array|Async|MipmappedArray)?)|"
    r"cu(?:Mem(?:Alloc(?:Async|FromPoolAsync|Managed|Pitch)*|Free(?:Async)?|Create|Release)|"
    r"(?:Array|Array3D|MipmappedArray)(?:Create|Destroy)))"
    r"(?=$|_v\d+|_pt[ds][sz])"
)
GRAPH_ALLOC = re.compile(r"^(?:cuda|cu)Graph(?:AddMem(?:Alloc|Free)Node|MemTrim)")
MAPPING = re.compile(
    r"^(?:cudaIpc(?:Get|Open|Close)MemHandle|"
    r"cu(?:Ipc(?:Get|Open|Close)MemHandle|Mem(?:Address(?:Reserve|Free)|Map(?:ArrayAsync)?|"
    r"Unmap|ExportToShareableHandle|ImportFromShareableHandle|SetAccess)|Multicast\w+))"
)
HOST = re.compile(
    r"^(?:cuda(?:MallocHost|HostAlloc|FreeHost|HostRegister|HostUnregister)|"
    r"cuMem(?:AllocHost|FreeHost|HostAlloc|HostRegister|HostUnregister))"
)
LAUNCH = re.compile(r"^(?:cuda|cu)Graph(?:Instantiate|Launch|Upload|ExecDestroy|Destroy)")


def pid(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def table_names(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}


def phase_at(markers: dict[str, int], instant: int) -> str:
    phase = "startup"
    for label, time in markers.items():
        if instant < time:
            break
        phase = label
    return phase


def external_stack(db: sqlite3.Connection, tables: set[str], callchain: int | None) -> bool:
    if callchain is None or "CUDA_CALLCHAINS" not in tables:
        return False
    frames = db.execute(
        """
        SELECT symbol.value, module.value FROM CUDA_CALLCHAINS AS chain
        JOIN StringIds AS symbol ON chain.symbol = symbol.id
        JOIN StringIds AS module ON chain.module = module.id
        WHERE chain.id = ?
    """,
        (callchain,),
    )
    return any("libnccl.so" in module and module != "" for _, module in frames)


def classify(
    db: sqlite3.Connection,
    tables: set[str],
    name: str,
    callchain: int | None,
) -> str | None:
    if ALLOC.match(name) or GRAPH_ALLOC.match(name):
        return "nccl" if external_stack(db, tables, callchain) else "device-api"
    if MAPPING.match(name):
        return "mapping"
    if HOST.match(name):
        return "host"
    if LAUNCH.match(name):
        return "graph"
    return None


def cuda_apis(
    db: sqlite3.Connection,
    tables: set[str],
    children: dict[int, str],
) -> Iterator[ApiCall]:
    runtime = """
        SELECT api.*, str.value AS apiName
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS api JOIN StringIds AS str ON api.nameId = str.id
        WHERE api.globalTid IS NOT NULL
    """
    driver = """
        SELECT api.*, str.value AS apiName
        FROM CUPTI_ACTIVITY_KIND_DRIVER AS api JOIN StringIds AS str ON api.nameId = str.id
        WHERE api.globalTid IS NOT NULL
    """
    for table, query in (
        ("CUPTI_ACTIVITY_KIND_RUNTIME", runtime),
        ("CUPTI_ACTIVITY_KIND_DRIVER", driver),
    ):
        if table not in tables:
            continue
        cursor = db.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute(query)
        has_callchain = any(column[0] == "callchainId" for column in cursor.description)
        for row in cursor:
            if pid(row["globalTid"]) not in children:
                continue
            chain = row["callchainId"] if has_callchain else None
            kind = classify(db, tables, row["apiName"], chain)
            if kind:
                yield {
                    "start": row["start"],
                    "end": row["end"],
                    "thread": row["globalTid"],
                    "correlation": row["correlationId"],
                    "name": row["apiName"],
                    "status": row["returnValue"],
                    "kind": kind,
                }


def root_operations(apis: list[ApiCall]) -> Iterator[ApiCall]:
    enclosing = {}
    for call in sorted(apis, key=lambda api: (api["start"], -api["end"])):
        thread = call["thread"]
        stack = enclosing.setdefault(thread, [])
        while stack and stack[-1]["end"] < call["end"]:
            stack.pop()
        allocation_kinds = ("device-api", "nccl")
        if call["kind"] in allocation_kinds and stack and stack[-1]["kind"] in allocation_kinds:
            continue
        stack.append(call)
        yield call


def inventory(db: sqlite3.Connection, children: dict[int, str], phases: list[str]) -> Inventory:
    tables = table_names(db)
    assert {"NVTX_EVENTS", "CUDA_GPU_MEMORY_USAGE_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"} <= tables
    markers = {
        text.removeprefix(MARKER): start
        for text, start in db.execute(
            """
        SELECT COALESCE(event.text, str.value), event.start FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS str ON event.textId = str.id
        WHERE COALESCE(event.text, str.value) LIKE ? ORDER BY event.start
    """,
            (MARKER + "%",),
        )
    }
    assert list(markers) == phases, markers
    apis = list(cuda_apis(db, tables, children))
    roots = list(root_operations(apis))
    resolved = {}
    for call in apis:
        key = (pid(call["thread"]), call["correlation"])
        previous = resolved.get(key)
        if previous is None or call["start"] >= previous["start"]:
            resolved[key] = call
    memory = []
    for row in db.execute("""
        SELECT e.start, e.globalPid, e.deviceId, e.address, e.bytes, e.correlationId,
               k.name, op.name FROM CUDA_GPU_MEMORY_USAGE_EVENTS AS e
        JOIN ENUM_CUDA_MEM_KIND AS k ON e.memKind = k.id
        JOIN ENUM_CUDA_DEV_MEM_EVENT_OPER AS op ON e.memoryOperationType = op.id
        ORDER BY e.start
    """):
        start, process, device, address, size, correlation, kind, operation = row
        child = pid(process)
        if child not in children:
            continue
        api = resolved.get((child, correlation))
        ownership = api["kind"] if api else "unattributed"
        if any(kind.endswith("_" + name) for name in ("PINNED", "PAGEABLE", "HOST")):
            ownership = "host"
        elif kind in (
            "CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC",
            "CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC",
        ):
            ownership = "module-static"
        memory.append(
            {
                "phase": phase_at(markers, start),
                "role": children[child],
                "device": device,
                "address": address,
                "bytes": size,
                "kind": kind,
                "operation": operation,
                "api": api["name"] if api else None,
                "ownership": ownership,
            }
        )
    calls = Counter()
    for api in roots:
        role = children[pid(api["thread"])]
        phase = phase_at(markers, api["start"])
        calls[(phase, role, api["kind"], api["name"], api["status"])] += 1
    return {
        "markers_ns": markers,
        "api_counts": [
            {
                "phase": phase,
                "role": role,
                "kind": kind,
                "api": api,
                "status": status,
                "count": count,
            }
            for (phase, role, kind, api, status), count in sorted(calls.items())
        ],
        "memory_events": memory,
    }


def verify(summary: Inventory, world: int) -> None:
    device_events = [
        e
        for e in summary["memory_events"]
        if e["ownership"] not in ("host", "mapping", "nccl", "module-static")
    ]
    startup = {
        e["device"]
        for e in device_events
        if e["role"] == "server" and e["phase"] == "startup" and e["bytes"]
    }
    assert len(startup) == world, ("Missing per-rank startup memory positive control", startup)
    calls = [c for c in summary["api_counts"] if c["role"] == "server"]
    assert any(c["phase"] == "startup" and c["kind"] == "device-api" for c in calls), calls
    assert any(c["phase"] == "shutdown" and c["kind"] == "device-api" for c in calls), calls
    assert any(
        c["phase"] == "reload" and c["kind"] == "mapping" and "Ipc" in c["api"]
        for c in summary["api_counts"]
        if c["role"] == "daemon"
    ), calls
    serving = set(summary["markers_ns"]) - {"shutdown"}
    unexpected = [
        c for c in summary["api_counts"] if c["kind"] == "device-api" and c["phase"] in serving
    ] + [e for e in device_events if e["phase"] in serving and e["bytes"]]
    assert not unexpected, unexpected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    children = {run["server_pid"]: "server", run["daemon_pid"]: "daemon"}
    with sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True) as db:
        result = inventory(db, children, run["phases"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    verify(result, run["tp"])


if __name__ == "__main__":
    main()
