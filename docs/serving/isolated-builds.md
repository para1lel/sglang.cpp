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

```bash
export XMAKE_CONFIGDIR="$work/config" XMAKE_GLOBALDIR="$work/global"
export XMAKE_BUILDDIR="$work/build" XMAKE_TMPDIR="$work/tmp"
export TMPDIR="$work/tmp" XDG_CONFIG_HOME="$work/config"
export CUDA_CACHE_PATH="$work/cuda-cache"
cd -- "$snapshot"
test "$(pwd -P)" = "$snapshot"
xmake f -P "$snapshot" -F "$snapshot/xmake.lua" -o "$work/build" -m release -p linux -a x86_64
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
