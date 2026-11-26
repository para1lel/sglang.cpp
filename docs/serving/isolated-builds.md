# Isolated XMake builds

For host checks, run `bash tools/build.sh` from the source snapshot. Its
[build probe](../../tools/build-probe.sh#L24)<!-- prefix: "if [[ -n \"$S" --> uses fresh scratch
by default. `--build-dir` retains a build under an identity derived from the
checkout, toolchain and lockfile. The probe prints the work directory and hashes
the outputs.

For a Release server build, leave the shared checkout read-only. Export the
selected source tree to a private snapshot and enter that snapshot inside locked
Pixi. Use absolute, canonical `$work` and `$snapshot` paths and create the private
directories before invoking XMake.

After applying a patch, verify the resulting files from the private snapshot root
before configuring XMake. Use `sha256sum --check` with an independent manifest of
the intended contents of every added or modified file. List paths relative to the
snapshot root. For each expected deletion, check `test ! -e "$path" && test ! -L "$path"`.
Stop if any check fails.

XMake uses a checksum-pinned source archive for CUTLASS v4.9.0dev at commit `f8acf1aa78f96ff2f1ba7662fe495e5535c461bf`. The shared SM100 prefill projection compiles with the CUTLASS/CuTe headers. The server has no CUTLASS runtime library dependency. [Package pin](../../xmake.lua#L16)<!-- prefix: "add_versions" --> · [SM100 binding](../../xmake.lua#L106)<!-- prefix: "add_packages" -->

The wrapper uses private XMake repository mode and installs from a local CUTLASS archive. Without an archive override, it reuses `$SGLANG_HOST_ROOT/cache/cutlass/<sha256>/cutlass-4.9.0dev.tar.gz` after verifying its hash. It fetches the pinned archive if the cached file is missing or fails verification. [Archive cache](../../tools/build-probe.sh#L35)<!-- prefix: "cached_cutla" --> · [Private configure](../../tools/build-probe.sh#L83)<!-- prefix: "\"$xmake\" g -" -->

Set `SGLANG_CUTLASS_ARCHIVE` to an absolute path to supply the original archive to the wrapper or the direct recipe below. XMake verifies the archive hash. For a network-free direct configure, run `xmake g --network=private` after exporting `XMAKE_GLOBALDIR` and before `xmake f`. Record the archive hash with the build evidence. [Wrapper forwarding](../../tools/build.sh#L42)<!-- prefix: "SGLANG_CUTLA" --> · [Archive override](../../xmake.lua#L9)<!-- prefix: "local archiv" -->

```bash
export XMAKE_CONFIGDIR="$work/config" XMAKE_GLOBALDIR="$work/global"
export XMAKE_BUILDDIR="$work/build" XMAKE_TMPDIR="$work/tmp"
export TMPDIR="$work/tmp" XDG_CONFIG_HOME="$work/config"
export CUDA_CACHE_PATH="$work/cuda-cache"
cd -- "$snapshot"
test "$(pwd -P)" = "$snapshot"
xmake f -y -P "$snapshot" -F "$snapshot/xmake.lua" -o "$work/build" -m release -p linux -a x86_64
xmake build -P "$snapshot" -F "$snapshot/xmake.lua" -v sglang-server
binary=$(realpath -e "$work/build/bin/sglang-server")
test "$binary" = "$work/build/bin/sglang-server"
sha256sum "$snapshot/xmake.lua" "$binary"
```

Retain the source tree or patch identity, Pixi lock hash, effective directory and
arguments, private configuration files, verbose compile and link log, and
resolved executable hash. Check that the configured output and executable path
belong to the private work directory. A fresh identity requires evidence of
compilation and linking. A successful no-op can refer to an earlier build or an
enclosing project. Compare the shared checkout status and index before and after
the build when recording read-only custody.
