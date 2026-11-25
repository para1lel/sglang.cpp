"""Reconcile each native daemon's acknowledged cold transfers with the joint receipt."""

import cold_cache
from saved_evidence import require


FAMILIES = ("gqa", "gdn", "draft")
COUNTERS = ("stores", "restores", *(f"{f}_{d}_bytes" for f in FAMILIES
                                  for d in ("read", "written")))


def rank_name(rank, counter):
    return f"sglang_cpp:lmcache_rank{rank}_{counter}_total"


def replay(receipt, aggregate, previous):
    names = tuple(rank_name(rank, counter) for rank in (0, 1) for counter in COUNTERS)
    samples = {key: cold_cache.parse_metrics(receipt[key], names) for key in aggregate}
    before, cleared, after = (samples[key] for key in ("before", "cleared", "after"))
    require((previous is None or before == previous) and before == cleared,
            "Rank-local counters changed outside the drained cohort")
    stores = cold_cache.LMCACHE_STORES
    expected = aggregate["after"][stores] - aggregate["cleared"][stores]
    delta = {name: after[name] - cleared[name] for name in names}
    for rank in (0, 1):
        require(expected > 0 and delta[rank_name(rank, "stores")] == expected,
                f"Rank {rank} daemon did not acknowledge every checkpoint store")
        require(delta[rank_name(rank, "restores")] == 0, f"Rank {rank} restored a cold checkpoint")
    for family in FAMILIES:
        for direction in ("read", "written"):
            counter = f"{family}_{direction}_bytes"
            a, b = (delta[rank_name(rank, counter)] for rank in (0, 1))
            group = f"sglang_cpp:lmcache_{counter}_total"
            total = aggregate["after"][group] - aggregate["cleared"][group]
            require(a + b == total, f"{counter}: rank and joint byte counts disagree")
            if direction == "read":
                require(a == b == 0, f"{counter}: a rank issued a cold restore")
            elif family == "draft":
                require(a > 0 and b == 0, "Only rank 0 stores draft state")
            else:
                require(a == b and a > 0, f"{counter}: expected both equal target shards")
    return after
