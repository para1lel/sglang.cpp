# Preserve resource lifetimes

Read this reference for asynchronous reads, native handles, shared buffers, snapshots, graphs, and cleanup.
Apply the backend's versioned synchronization contract.
Use supplied domain guidance for instruction-specific fences and completion rules when relevant.

## Define ownership at each boundary

Describe publication, issue, completion, consumption, and reuse or release.
Identify each resource's owner and actual users.
Separate host context, stream, descriptor, storage, request, graph, and replay lifetimes.
Allocation, launch, wrapper destruction, or an event's existence does not establish completion.

Protect a source until every asynchronous reader finishes.
Use dependencies, immutable staging, or another mechanism that preserves the required captured value.
A destination allocation does not permit mutation of a source still being copied.
Keep shared immutable state shareable.
Give mutable state a private owner or use copy-on-write before a fork modifies it.

Publish reusable state only after its data and dependencies are valid.
Coordinate cancellation, eviction, and closure with already admitted submissions.
Retire pending users before storage is recycled.
When completion cannot be proved, retain the resource conservatively and report the reclamation limit.

## Preserve control and execution context

Complete shared control reads before later writes can replace their source values.
Examine loop bounds, predicates, macro arguments, and returns to uniform work as related sites.
Classify uniform entry separately from divergence within the body.
Place a collective synchronization point where every required participant can reach it.
Use an enclosing uniform join when an inner scope is divergent.

Establish the required driver context on each host thread before descriptor encoding or launch.
Restore temporary host state after both success and failure.
Keep the owning context available during deferred cleanup.
Examine logical views and descriptor metadata before using raw pointers.
Include output views and empty-work paths.

## Make setup and cleanup transactional

Give native resources an owner during setup, before public handle publication.
Unwind partially acquired resources when later setup fails.
Keep the complete old allocation set until every replacement is acquired and published.
Publish replacements coherently so consumers cannot observe a partial new set.

Remove an allocation from its owner only after release succeeds.
Retain failed and unattempted releases for later cleanup.
Preserve the original error while reporting cleanup failures.
Keep retries compatible with the resource's actual state.
Do not drop handles merely because their wrapper is closing.

Track graph ownership separately from each submitted replay.
Retain resources while the graph can reference them and until in-flight replays complete.
An eager execution alone cannot establish captured-buffer validity.
Inputs, plans, backend changes, and cancellation can affect replay ownership.

## Request evidence for the changed invariant

Give test the changed ownership transition and the failure it must distinguish.
Useful checks can include partial acquisition, allocation growth, release failure, concurrent closure, cancellation, and replay.
Use fault injection when ordinary execution cannot reach a cleanup failure.
Select representative transitions that expose distinct invariants.
Numerical agreement alone cannot prove race freedom or safe reclamation.
A race probe must preserve the shared address and conflicting accesses.
Return unresolved ordering and lifetime assumptions with the candidate identity.
