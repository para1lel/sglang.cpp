# Establish source and execution identity

Read this reference for build, cache, packaging, dispatch, and observer changes.
Coding owns the implementation mechanisms.
Test independently examines the resulting identities and execution claims.

## Pin the inputs and outputs

Associate a successful build with fixed source, applied patches, effective options, dependency identities, and output artifact hashes.
Keep the expected identities fixed before validation.
Computing new hashes for changed inputs does not verify the earlier evidence.
Validate copied source, archives, and other inputs against their recorded expected hashes before use.
Publish a success association only after the build and required probes succeed.
Keep incomplete or failed builds distinct from successful artifacts.

Distinguish the library's identity from the executable that consumes it.
An available symbol establishes capability, not the exact source or options used to produce a binary.
Record the executable, loaded library, and relevant build inputs separately.
Inspect the effective loaded or imported path.
A matching package name or another checkout's passing run is insufficient.

Treat external archives as actual link dependencies.
Changes to an archive must invalidate the consuming artifact when its contents affect linking.
Verify static registration, archive order, and relevant linker behavior in the emitted command.
Examine mixed host and device link paths separately.
A declared link group does not establish that every compiler driver emits it.

## Make cache identity complete

Include transitive source and header dependencies, effective compiler options, and preprocessing decisions.
Capture optional-header absence as well as presence.
Include header search resolution when it can select a different source.
Capture effective configuration once for dependency discovery and compilation.
Recheck dependency and preprocessing identity before publishing the result.

Treat malformed cache metadata as unavailable evidence.
Avoid accepting stale binaries because their paths still exist.
Use isolated installed-package checks when they can expose missing resources or source shadowing.
Have main assign cache cleanup to the appropriate owner and a bounded location.
Preserve unrelated environments and caches.

For source-introspecting compilation or collection, keep source bytes stable through the operation.
Bind generated code and configuration to the same input snapshot.
Record changed submodules, code generators, and relevant tool versions.
An ignored local directory alone is not a recoverable source record.

## Establish the production path

Trace a public request through effective dispatch to the changed branch.
Check resolved settings rather than requested settings alone.
Use the production configuration for production claims.
Include initialization, registration, feature policy, and fallback conditions in the path analysis.
A linked feature or successful isolated helper call does not establish that the production branch executed.

Give each reference and observation an explicit scope.
An independent mathematical reference can establish numerical meaning.
A pinned application reference can establish compatible application behavior.
A simple verified runtime can isolate state or dispatch differences.
Shared defective code limits a comparison's independence.

Keep candidate mutations separate from reference outputs and validation inputs.
Preserve required aliases and offsets within independent input sets.
Return evidence requests to test when source analysis cannot establish the complete path.
Source reachability can be proved statically for a closed path.
Runtime-dependent routing needs an observation that distinguishes the selected branch.

## Make observers reachable

Trace the production call into the actual observer or interception boundary.
Check compiled symbols and emitted launch entries when compilation can change that boundary.
Use matching toolkit handle types and name-lookup APIs.
Source names and manual wrapper calls do not establish interception of compiled launches.

Require a positive control that reaches the measured path.
For a compiled launch observer, use a separately compiled launch through that boundary.
Have test verify that the control creates an observation or expected rejection.
Only then interpret zero counters as possible absence of work.
Keep guards, counters, and diagnostics from changing the workload they claim to observe.

An incomplete observer invalidates its affected observation claim.
It does not automatically invalidate independent numerical evidence.
Identify the affected conclusions and repeat the necessary observation with fixed inputs.
Do not replace expected evidence after a failure merely to accept the current output.
