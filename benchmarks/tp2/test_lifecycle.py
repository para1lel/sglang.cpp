"""Regression checks for the one-process, two-device native topology."""

import pytest

import cache_activity
import cold_cache
from native import graph_policy, owners, rank_table


DEVICES = [f"GPU-00000000-0000-0000-0000-00000000000{n}" for n in (1, 2)]


def rank_line(rank):
    return (f"tp2 rank={rank} device={rank} uuid={DEVICES[rank][4:]} pid=31 nccl=ok "
            f"weights=100 lm_head={1-rank} mtp={1-rank}")


def test_native_world_matches_both_local_ranks_and_draft_owner():
    text = "\n".join((rank_line(1), "tp2 world=2 ranks=2", rank_line(0)))
    ranks = rank_table(text, DEVICES, 31)
    assert [r["pid"] for r in ranks] == [31, 31]
    assert [r["mtp"] for r in ranks] == [True, False]
    for bad in (text.replace("pid=31", "pid=32", 1), text.replace("lm_head=0", "lm_head=1"),
                text + "\n" + rank_line(0), text.replace(rank_line(1), "")):
        with pytest.raises(ValueError):
            rank_table(bad, DEVICES, 31)


def test_graph_policy_records_real_boolean_and_requires_shared_setting(tmp_path):
    for value in (True, False):
        assert graph_policy({"batch_cuda_graphs": value}, tmp_path) is value
    for info in ({}, {"batch_cuda_graphs": 0}, {"batch_cuda_graphs": "false"}):
        with pytest.raises(ValueError, match="Boolean"):
            graph_policy(info, tmp_path)
    run = tmp_path / "W1/native-k4"
    run.mkdir(parents=True)
    (run / "ready.json").write_text('{"config":{"batch_cuda_graphs":false}}')
    with pytest.raises(ValueError, match="changed across"):
        graph_policy({"batch_cuda_graphs": True}, tmp_path)


def test_joint_gpu_ownership_includes_one_server_and_exactly_the_local_daemon(tmp_path):
    inventory = tmp_path / "compute.csv"
    rows = [f"{gpu}, {pid}, owned, 50 MiB" for gpu, pids in zip(DEVICES, [(31, 41), (31, 42)])
            for pid in pids]
    inventory.write_text("\n".join(rows))
    owners(inventory, DEVICES, [41, 42], 31)
    for changed in (rows[:-1], rows + rows[-1:], rows + [rows[0].replace(", 31,", ", 99,")],
                    rows[:-1] + [rows[-1].replace(", 42,", ", 41,")]):
        inventory.write_text("\n".join(changed))
        with pytest.raises(ValueError, match="ownership"):
            owners(inventory, DEVICES, [41, 42], 31)


@pytest.mark.parametrize("fault", [None, "idle-rank", "wrong-draft-owner", "partial-shard",
                                    "cold-read", "between-cohorts"])
def test_rank_cache_deltas_reconcile_each_family_on_both_daemons(fault):
    names = cache_activity.rank_name
    baseline = {names(r, c): 0 for c in cache_activity.COUNTERS for r in (0, 1)}
    active = baseline.copy()
    for rank in (0, 1):
        for counter, amount in (("stores", 2), ("gqa_written_bytes", 128),
                                ("gdn_written_bytes", 512), ("draft_written_bytes", 64)):
            if rank == 0 or counter != "draft_written_bytes":
                active[names(rank, counter)] = amount
    if fault == "idle-rank":
        active[names(1, "stores")] = 0
    if fault == "wrong-draft-owner":
        active[names(1, "draft_written_bytes")] = 1
    if fault == "partial-shard":
        active[names(0, "gqa_written_bytes")] += 1
    if fault == "cold-read":
        active[names(1, "gdn_read_bytes")] = 1

    def text(values):
        return "\n".join(f'{name}{{engine="0"}} {value}' for name, value in values.items())

    receipt = {"before": text(baseline), "cleared": text(baseline), "after": text(active)}
    after = {cold_cache.LMCACHE_STORES: 2}
    for family in cache_activity.FAMILIES:
        for direction in ("read", "written"):
            counter = f"{family}_{direction}_bytes"
            after[f"sglang_cpp:lmcache_{counter}_total"] = sum(active[names(r, counter)]
                                                               for r in (0, 1))
    zero = dict.fromkeys(after, 0)
    aggregate = {"before": zero, "cleared": zero, "after": after}
    if fault is None:
        assert cache_activity.replay(receipt, aggregate, None) == active
    else:
        with pytest.raises(ValueError):
            previous = active if fault == "between-cohorts" else None
            cache_activity.replay(receipt, aggregate, previous)
