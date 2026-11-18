#!/usr/bin/env bash
set -euo pipefail

pin=81a8cb063e6247fe254d7caf7e98a6691d71022a
version=0.34.0rc1.dev787+g81a8cb063
root=/data1/shared/dongwu.chen/environments/vllm-${pin:0:12}
here=$(cd "$(dirname "$0")" && pwd)
venv=$root/venv
python=$venv/bin/python
name=vllm-$version-cp38-abi3-manylinux_2_28_x86_64.whl
index=https://wheels.vllm.ai/$pin/cu132/vllm/
url=https://wheels.vllm.ai/$pin/${name/+/%2B}
torch_name=torch-2.14.0+cu132-cp312-cp312-manylinux_2_28_x86_64.whl
torch_url=https://download-r2.pytorch.org/whl/cu132/${torch_name/+/%2B}
export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy ALL_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export CUDA_VISIBLE_DEVICES="" UV_CONCURRENT_DOWNLOADS=4
unset PYTHONPATH PYTHONHOME

mkdir -p "$root/wheels"
exec 9>"$root/setup.lock"
flock 9
if [[ ! -d $root/source/.git ]]; then
    git init -q "$root/source"
fi
if ! git -C "$root/source" cat-file -e "$pin^{commit}" 2>/dev/null; then
    git -C "$root/source" fetch --no-tags --depth=1 https://github.com/vllm-project/vllm.git "$pin"
    test "$(git -C "$root/source" rev-parse FETCH_HEAD)" = "$pin"
fi
if ! git -C "$root/source" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$root/source" checkout --detach "$pin"
fi
test "$(git -C "$root/source" rev-parse HEAD)" = "$pin"
git -C "$root/source" diff --quiet HEAD
curl -fsSL --retry 2 --max-time 45 "${index}metadata.json" -o "$root/index.json"
jq -e --arg name "$name" 'any(.[]; .filename == $name)' "$root/index.json" >/dev/null
fetch_wheel() {
    local file=$root/wheels/$1
    if [[ ! -f $file ]]; then
        curl -fLsS --retry 3 --max-time 600 "$2" -o "$file.part"
        mv "$file.part" "$file"
    fi
}
fetch_wheel "$name" "$url"
fetch_wheel "$torch_name" "$torch_url"
(cd "$root/wheels" && sha256sum --check "$here/wheel.sha256")
if [[ ! -x $python ]]; then
    uv venv --python /usr/bin/python3.12 "$venv"
fi
test "$("$python" -I -c 'import sys; print(sys.prefix)')" = "$venv"
uv pip install --python "$python" -c "$here/constraints.txt" \
    "$root/wheels/$name" "$root/wheels/$torch_name"
uv pip check --python "$python"
"$python" -I -c 'import sys, vllm; assert sys.argv[1] == vllm.__version__' "$version"
printf '%s\n' "$pin" "$version" "$url" >"$root/artifact.txt"
cp "$here/wheel.sha256" "$root/wheel.sha256"
uv pip freeze --python "$python" >"$root/freeze.txt"
site=$("$python" -I -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
(cd "$site" && find vllm -name '*.so' -type f -print0 | sort -z | xargs -0r sha256sum) \
    >"$root/native.sha256"
test -s "$root/native.sha256"
