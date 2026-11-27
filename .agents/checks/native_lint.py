import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from function_spans import HEADER_SUFFIXES
from native_tokens import DIRECTIVE_IDENTIFIER, SPLICE
from tool_process import invoke as invoke_tool

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}
CUDA_SUFFIXES = {".cu", ".cuh"}
SM100A_SOURCES = {
    "src/prefill_fp8_sm100.cu",
    "src/runtime/backend/mnnvl_ar.cu",
    "src/text/decode_proj_bf16.cu",
    "src/text/decode_proj_fp8.cu",
    "src/text/decode_proj_host.cu",
    "src/text/decode_proj_sm100.cu",
    "src/text/draft_depth.cu",
    "src/text/draft_seed_fc_sm100.cu",
    "src/text/gqa/decode_layout.cu",
    "src/text/gqa/device_geometry.cu",
    "src/text/gqa/prefill_common.cu",
    "src/text/gqa/tmem_ops.cu",
    "src/text/gqa/tmem_staged.cu",
    "src/text/gqa/tmem_staged_kernels.cu",
    "src/text/gqa_decode_long.cu",
    "src/text/gqa_decode_split.cu",
    "src/text/mxfp8_gqa_prefill_tmem.cu",
    "src/text/prefill_down_sm100.cu",
    "src/text/prefill_gdn_gates_sm100.cu",
    "src/text/tc_vocab.cu",
}
GQA_DEVICE_SOURCES = {
    "src/text/gqa/decode_layout.cu",
    "src/text/gqa/device_geometry.cu",
    "src/text/gqa/prefill_common.cu",
}
MEDIA_SOURCES = {"apps/server/media_mp4.cpp", "apps/server/media_png.cpp"}
PROBES = ("tools/toolchain-probe.cpp", "tools/toolchain-probe.cu")
BLOCK_COMMENT = rb"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/"
INCLUDE_SPACE = rb"(?:[ \t\v\f]|" + BLOCK_COMMENT + rb")*"
INCLUDE_TOKEN = re.compile(
    rb"^"
    + INCLUDE_SPACE
    + rb"(?:#|%:)"
    + INCLUDE_SPACE
    + rb"include(?!"
    + DIRECTIVE_IDENTIFIER
    + rb")"
    + INCLUDE_SPACE
    + rb"""(?:"(?P<quoted>[^"\r\n]+)"|<(?P<angled>[^>\r\n]+)>|(?P<macro>[^/"<\s]))"""
    + rb'|R"(?P<delimiter>[^ ()\\\t\v\f\r\n]{0,16})\(.*?\)(?P=delimiter)"'
    + rb"|"
    + BLOCK_COMMENT
    + rb"|//[^\r\n]*"
    + rb"""|"(?:\\[^\r\n]|[^"\\\r\n])*"|'(?:\\[^\r\n]|[^'\\\r\n])*'""",
    re.MULTILINE | re.DOTALL,
)
TRACE = re.compile(r"(?m)^\.+ (.+)$\n?")
DIAGNOSTIC = re.compile(
    r"(?m)^(.+?):(\d+):(\d+): (?:warning|(?:fatal )?error): (.*?)"
    r"(?: \[([^\]\n]+)\])?$",
)
TOOL_ERROR = re.compile(r"(?m)^(?:fatal )?error: ")
CUTLASS_PACKAGE = re.compile(
    r'package\("sglang-cutlass-headers"\).*?add_versions\("([\w.]+)", "([\da-f]{64})"\)',
    re.DOTALL,
)
CUTLASS_MISSING = re.compile(r"'(?:cutlass|cute)/[^']+' file not found")
NVCC_HINT_WARNING = (
    "clang-diagnostic-unknown-attributes",
    "unknown attribute 'nv_inline_hint' ignored",
)

LineSelection = dict[str, list[tuple[int, int]] | None]
IncludeParents = defaultdict[str, set[str]]
FindingKey = tuple[str, int | None, int, str, str]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    message: str
    check: str
    output: str

    @property
    def critical(self) -> bool:
        return self.check.startswith("clang-analyzer-")

    @property
    def compiler_error(self) -> bool:
        return not self.check or self.check == "clang-diagnostic-error"

    def key(self, line: int | None) -> FindingKey:
        return self.path, line, self.column, self.check, self.message


def tool_output(arguments: list[str], root: Path) -> tuple[int, str]:
    result = invoke_tool(arguments, root, merge_stderr=True, timeout=120)
    return result.returncode, result.stdout.decode(errors="replace")


def include_directives(source: bytes) -> Iterator[tuple[bytes, bytes]]:
    source = re.sub(SPLICE, b"", source.removeprefix(b"\xef\xbb\xbf"))
    for match in INCLUDE_TOKEN.finditer(source):
        if match["quoted"] is not None:
            yield b'"', match["quoted"]
        elif match["angled"] is not None:
            yield b"<", match["angled"]
        elif match["macro"] is not None:
            yield b"", b""


def include_parents(root: Path) -> tuple[IncludeParents, set[str]]:
    parents: IncludeParents = defaultdict(set)
    macro_owners = set()
    for owner in root.rglob("*"):
        if owner.suffix.lower() not in SOURCE_SUFFIXES | HEADER_SUFFIXES or not owner.is_file():
            continue
        relative = owner.relative_to(root).as_posix()
        for quote, name in include_directives(owner.read_bytes()):
            if not quote:
                macro_owners.add(relative)
                continue
            directories = (owner.parent, root / "include") if quote == b'"' else (root / "include",)
            for directory in directories:
                candidate = (directory / os.fsdecode(name)).resolve()
                if candidate.is_relative_to(root) and candidate.is_file():
                    parents[candidate.relative_to(root).as_posix()].add(relative)
                    break
    return parents, macro_owners


def consumers(root: Path, headers: set[str]) -> set[str]:
    parents, macro_owners = include_parents(root)
    seen = headers | macro_owners
    pending = list(seen)
    units = set()
    while pending:
        name = pending.pop()
        if Path(name).suffix.lower() in SOURCE_SUFFIXES:
            units.add(name)
        for parent in parents[name] - seen:
            seen.add(parent)
            pending.append(parent)
    return units


def toolchain_flags(prefix: Path) -> list[str]:
    target = "x86_64-conda-linux-gnu"
    sysroot = prefix / target / "sysroot"
    if sysroot.is_dir() and (prefix / "lib/gcc" / target).is_dir():
        return [f"--target={target}", f"--gcc-toolchain={prefix}", f"--sysroot={sysroot}"]
    return []


def package_includes(prefix: Path) -> list[str]:
    sdk = prefix / "targets/x86_64-linux/include"
    nccl = sorted(prefix.glob("lib/python*/site-packages/nvidia/nccl/include"))
    includes = [sdk, sdk / "cccl", *nccl, prefix / "include"]
    if media := os.environ.get("SGLANG_LINT_MEDIA_INCLUDE"):
        includes.append(Path(media).resolve())
    return system_includes(includes)


def system_includes(includes: list[Path]) -> list[str]:
    return [
        option
        for directory in includes
        if directory.is_dir()
        for option in ("-isystem", str(directory))
    ]


def cutlass_archive(root: Path, prefix: Path) -> tuple[Path, str]:
    spec = CUTLASS_PACKAGE.search((root / "xmake.lua").read_text(encoding="utf-8"))
    if spec is None:
        raise ValueError("Cannot read the pinned CUTLASS version and SHA-256 from xmake.lua.")
    version, checksum = spec.groups()
    if archive := os.environ.get("SGLANG_CUTLASS_ARCHIVE"):
        return Path(archive).resolve(), checksum
    manifest = os.environ.get("PIXI_PROJECT_MANIFEST")
    pixi = prefix.parent.parent
    default_root = pixi.parent if pixi.name == ".pixi" else prefix
    host_root = Path(manifest).parent if manifest else default_root
    host_root = Path(os.environ.get("SGLANG_HOST_ROOT", host_root))
    return host_root / "cache/cutlass" / checksum / f"cutlass-{version}.tar.gz", checksum


def extract_cutlass(archive: Path, checksum: str, destination: Path) -> list[Path]:
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing offline CUTLASS archive {archive}. Set SGLANG_CUTLASS_ARCHIVE to the "
            "verified build archive or SGLANG_LINT_CUTLASS_INCLUDE to its installed headers."
        )
    with archive.open("rb") as handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(
                f"CUTLASS archive SHA-256 mismatch for {archive} (expected {checksum})."
            )
        handle.seek(0)
        with tarfile.open(fileobj=handle, mode="r|gz") as source:
            for member in source:
                _, _, relative = member.name.partition("/")
                if member.isfile() and relative.startswith(("include/", "tools/util/include/")):
                    source.extract(member, destination, filter="data")
    return [*destination.glob("*/include"), *destination.glob("*/tools/util/include")]


def cutlass_includes(root: Path, prefix: Path, temporary: Path) -> list[str]:
    if include := os.environ.get("SGLANG_LINT_CUTLASS_INCLUDE"):
        directory = Path(include).resolve()
        if not (directory / "cutlass/cutlass.h").is_file():
            raise FileNotFoundError(
                f"SGLANG_LINT_CUTLASS_INCLUDE lacks cutlass/cutlass.h: {directory}"
            )
        return system_includes([directory])
    archive, checksum = cutlass_archive(root, prefix)
    return system_includes(extract_cutlass(archive, checksum, temporary))


def cuda_flags(name: str, prefix: Path) -> tuple[list[str], ...]:
    arch = "sm_100a" if name in SM100A_SOURCES or name.lower().endswith(".cuh") else "sm_100"
    flags = [
        "-std=c++17",
        "-x",
        "cuda",
        f"--cuda-path={prefix}",
        "-Wno-unknown-cuda-version",
    ]
    if name == "tests/decoder/gdn_check.cu":
        flags.extend(("-DSGLANG_TEST_GDN_TILED", "-DSGLANG_TEST_GDN_CHUNK"))
    main = [*flags, "--cuda-compile-host-device", f"--cuda-gpu-arch={arch}"]
    if name in GQA_DEVICE_SOURCES:
        return main, [
            *flags,
            "--cuda-device-only",
            "--cuda-gpu-arch=sm_100",
            "-DSGLANG_GQA_DEVICE_ONLY",
        ]
    if name in {"src/text/rms_quantize.cu", "src/text/text_cuda_primitives.cu"}:
        return main, [
            *flags,
            "--cuda-device-only",
            "--cuda-gpu-arch=sm_100a",
            "-DSGLANG_TEXT_DEVICE_ONLY",
        ]
    return (main,)


def source_flags(name: str, prefix: Path) -> tuple[list[str], ...]:
    suffix = Path(name).suffix.lower()
    if suffix in CUDA_SUFFIXES:
        return cuda_flags(name, prefix)
    if suffix == ".c":
        return (["-std=c11", "-x", "c"],)
    flags = ["-std=c++17", "-x", "c++"]
    if name.startswith("src/"):
        flags.append("-DSGLANG_MNNVL_AR=1")
    if name == "apps/server/chat_media.cpp":
        return flags, [*flags, "-DSGLANG_MEDIA_CODECS"]
    if name in MEDIA_SOURCES:
        flags.append("-DSGLANG_MEDIA_CODECS")
    if name in {"src/media_preprocess.cpp", "src/media_resize.cpp"}:
        flags.append("-ffp-contract=off")
    return (flags,)


def project_path(root: Path, path: str) -> str:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = root / absolute
    return absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else path


def findings(root: Path, output: str) -> list[Finding]:
    output = TRACE.sub("", output)
    matches = list(DIAGNOSTIC.finditer(output))
    results = []
    for index, match in enumerate(matches):
        path, line, column, message, checks = match.groups()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(output)
        check = (checks or "").split(",", 1)[0]
        rendered = output[match.start() : end].replace(f"{root}/", "").rstrip()
        results.append(
            Finding(project_path(root, path), int(line), int(column), message, check, rendered)
        )
    return results


class ClangTidy:
    def __init__(
        self,
        root: Path,
        temporary: Path,
        files: LineSelection,
        baseline: Path | None,
    ) -> None:
        executable = shutil.which("clang-tidy")
        if executable is None:
            raise FileNotFoundError("Install clang-tidy from .agents/checks/requirements.txt.")
        self.root = root
        self.baseline = baseline
        self.files = files
        self.config = [executable, "--quiet", f"--config-file={root / '.clang-tidy'}"]
        self.prefix = Path(os.environ.get("CONDA_PREFIX", Path(executable).resolve().parents[1]))
        self.flags = toolchain_flags(self.prefix) + package_includes(self.prefix)
        self.wrapper = temporary / "header.cpp"
        self.cutlass: list[str] | None = None
        self.reported: set[FindingKey] = set()
        self.failed = False

    def verify(self) -> bool:
        status, output = tool_output([*self.config, "--verify-config"], self.root)
        if status:
            print(output.rstrip())
        return status == 0

    def analyze(
        self, root: Path, name: str, *, trace: bool = False
    ) -> tuple[list[Finding], set[str]]:
        path = root / name
        tidy = [*self.config]
        if path.suffix.lower() in CUDA_SUFFIXES:
            tidy.append("--checks=-misc-const-correctness")
        needs_cutlass = any(
            include.startswith((b"cutlass/", b"cute/"))
            for _, include in include_directives(path.read_bytes())
        )
        if path.suffix.lower() in HEADER_SUFFIXES:
            self.wrapper.write_text(
                f"#include {json.dumps(str(path), ensure_ascii=False)}\n", encoding="utf-8"
            )
            path = self.wrapper
        diagnostics = []
        included = set()
        options = ["-Wall", "-Wextra", f"-I{root / 'include'}", *self.flags]
        if trace:
            options.append("-H")
        for variant in source_flags(name, self.prefix):
            arguments = [*tidy, str(path), "--", *variant, *options]
            status, output = self.invoke(root, arguments, needs_cutlass=needs_cutlass)
            observed = findings(root, output)
            if status not in {0, 1} or TOOL_ERROR.search(output) or (status and not observed):
                message = f"clang-tidy failed for {name} (exit {status})"
                raise RuntimeError(f"{message}\n{output.rstrip()}")
            diagnostics.extend(observed)
            if trace:
                included.update(project_path(root, entry) for entry in TRACE.findall(output))
        return diagnostics, included

    def invoke(self, root: Path, arguments: list[str], *, needs_cutlass: bool) -> tuple[int, str]:
        if self.cutlass is None and not needs_cutlass:
            status, output = tool_output(arguments, root)
            if not CUTLASS_MISSING.search(output):
                return status, output
        if self.cutlass is None:
            self.cutlass = cutlass_includes(self.root, self.prefix, self.wrapper.parent)
        return tool_output([*arguments, *self.cutlass], root)

    def report(self, item: Finding) -> None:
        key = item.key(item.line)
        if key not in self.reported:
            print(item.output)
            self.reported.add(key)
        self.failed = True

    def inspect(self, name: str, *, trace: bool = False) -> set[str]:
        diagnostics, covered = self.analyze(self.root, name, trace=trace)
        delta = []
        for item in diagnostics:
            if (item.check, item.message) == NVCC_HINT_WARNING:
                continue
            if item.compiler_error or selected(item, self.files):
                self.report(item)
            elif item.critical:
                delta.append(item)
        if delta:
            self.inspect_delta(name, delta)
        return covered

    def inspect_delta(self, name: str, current: list[Finding]) -> None:
        old_findings = set()
        baseline = self.baseline
        if baseline is not None and (baseline / name).is_file():
            previous, _ = self.analyze(baseline, name)
            if not any(item.compiler_error for item in previous):
                old_findings = {item.key(item.line) for item in previous if item.critical}
        if not old_findings or baseline is None:
            for item in current:
                self.report(item)
            return
        locations = {}
        for item in current:
            if item.path not in locations:
                locations[item.path] = unchanged_lines(baseline, self.root, item.path)
            mapping = locations[item.path]
            line = item.line if mapping is None else mapping.get(item.line)
            if item.key(line) not in old_findings:
                self.report(item)


def unchanged_lines(original: Path, current: Path, name: str) -> dict[int, int] | None:
    if Path(name).is_absolute():
        return None
    old_path = original / name
    new_path = current / name
    if not old_path.is_file() or not new_path.is_file():
        return {}
    old_lines = old_path.read_bytes().splitlines()
    new_lines = new_path.read_bytes().splitlines()
    if old_lines == new_lines:
        return None
    matching = SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_matching_blocks()
    return {
        new + offset + 1: old + offset + 1 for old, new, size in matching for offset in range(size)
    }


def selected(item: Finding, files: LineSelection) -> bool:
    if item.path not in files:
        return False
    ranges = files[item.path]
    return ranges is None or any(start <= item.line <= end for start, end in ranges)


def inspect_selected(engine: ClangTidy) -> None:
    root, baseline, files = engine.root, engine.baseline, engine.files
    headers = {name for name in files if Path(name).suffix.lower() in HEADER_SUFFIXES}
    units = {name for name in files if name not in headers and (root / name).is_file()}
    if not files:
        units.update(name for name in PROBES if (root / name).is_file())
    uncovered = {name for name in headers if (root / name).is_file()}
    if uncovered:
        units.update(consumers(root, uncovered))
    if baseline is not None:
        previous = {name for name in headers if (baseline / name).is_file()}
        if previous:
            units.update(name for name in consumers(baseline, previous) if (root / name).is_file())
    for name in sorted(units):
        uncovered -= engine.inspect(name, trace=bool(uncovered))
    for header in sorted(uncovered):
        if header in uncovered:
            uncovered -= engine.inspect(header, trace=True)
            if header in uncovered:
                raise RuntimeError(f"clang-tidy could not confirm analysis of {header}.")


def run(root: Path, files: LineSelection, *, baseline: Path | None = None) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="sglang-clang-tidy-") as directory:
            engine = ClangTidy(root, Path(directory), files, baseline)
            if not engine.verify():
                return False
            inspect_selected(engine)
            return not engine.failed
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        tarfile.TarError,
    ) as error:
        print(f"Native lint error: {error}")
        return False
