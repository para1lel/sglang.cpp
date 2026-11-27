"""Reject cohorts whose server logs record serious CUDA driver launch failures."""

import re

DRIVER_LAUNCH_FAILURES = (
    "CUDA_ERROR_INVALID_HANDLE",
    "cudaErrorInvalidResourceHandle",
    "cudaErrorInvalidHandle",
    "CUDA_ERROR_LAUNCH_FAILED",
    "cudaErrorLaunchFailure",
    "CUDA_ERROR_LAUNCH_TIMEOUT",
    "cudaErrorLaunchTimeout",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "cudaErrorIllegalAddress",
    "CUDA_ERROR_MISALIGNED_ADDRESS",
    "cudaErrorMisalignedAddress",
    "CUDA_ERROR_ILLEGAL_INSTRUCTION",
    "cudaErrorIllegalInstruction",
    "CUDA_ERROR_HARDWARE_STACK_ERROR",
    "cudaErrorHardwareStackError",
    "CUDA_ERROR_ASSERT",
    "cudaErrorAssert",
)

_FAILED_LAUNCH_RECORD = re.compile(r"\blaunch failed: CUresult \d+ \(([^()]*)\)")
_MARKER_PATTERNS = tuple(
    re.compile("(?<![A-Za-z0-9_])" + re.escape(marker) + "(?![A-Za-z0-9_])")
    for marker in DRIVER_LAUNCH_FAILURES)


def driver_launch_failures(text):
    counts = {}
    for line in text.splitlines():
        record = _FAILED_LAUNCH_RECORD.search(line)
        if record and record.group(1) != "CUDA_SUCCESS":
            code = record.group(1)
            counts[code] = counts.get(code, 0) + 1
            continue
        for marker, pattern in zip(DRIVER_LAUNCH_FAILURES, _MARKER_PATTERNS):
            if pattern.search(line):
                counts[marker] = counts.get(marker, 0) + 1
    return counts


def require_healthy_driver_log(text, label):
    failures = driver_launch_failures(text)
    if failures:
        raise ValueError(f"{label}: server log records driver launch failures: {failures}")
