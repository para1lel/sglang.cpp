#!/bin/bash
set -euo pipefail

base=${SGLANG_HOST_ROOT:-/data1/shared/dongwu.chen}
version=v3.1.2
digest=ac216aa71f4954cc1129aabf1099571d618211c1ef8db8fba871377ec5fb47cd
upstream=https://github.com/xmake-io/xmake/releases/download
program="$base/software/xmake/$version/xmake"
command_path="$base/bin/xmake"

verify() {
    printf '%s  %s\n' "$digest" "$1" | sha256sum --check
}

case "$#:${1:-}" in
    0:) ;;
    1:--install)
        if [[ ! -e "$program" ]]; then
            mkdir -p -- "$(dirname -- "$program")"
            bundle=$(mktemp "${program}.XXXXXXXX")
            trap 'rm -f -- "$bundle"' EXIT
            proxy=${all_proxy:-${ALL_PROXY:-http://127.0.0.1:7890}}
            curl --fail --location --silent --show-error --retry 2 --proto '=https' \
                --proxy "${https_proxy:-${HTTPS_PROXY:-$proxy}}" --noproxy '' --output "$bundle" \
                "$upstream/$version/xmake-bundle-$version.linux.x86_64"
            verify "$bundle"
            chmod 755 "$bundle"
            ln -- "$bundle" "$program"
        fi
        verify "$program"
        if [[ ! -e "$command_path" && ! -L "$command_path" ]]; then
            ln -s -- "$program" "$command_path"
        fi
        ;;
    *) echo "Usage: bash tools/toolchain-xmake.sh [--install]" >&2; exit 2 ;;
esac

if [[ ! -x "$command_path" || "$(realpath -- "$command_path")" != "$program" ]]; then
    echo "Expected $command_path to select $program" >&2
    exit 1
fi
verify "$command_path"
