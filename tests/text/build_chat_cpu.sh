#!/usr/bin/env bash
set -euo pipefail

tests=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source=$(realpath -- "${1:?Pass the native loader source directory}")
build=${2:?Pass the matching Release build directory or - for a source build}
output=$(realpath -m -- "${3:?Pass the output executable path}")

sources=("$source/apps/sglang-server.cpp")
executor="$source/src/runtime/server/single_device_batch_executor.cpp"
if [[ -f "$executor" ]]; then
    sources+=("$executor")
fi
for name in batch_service completions flush_cache http loads text_request text_completion \
    chat_template chat_media chat_output chat_tool_constraints native_generate native_sessions \
    ollama_discovery ollama_generate \
    parse_function_call prefix_controls anthropic_count anthropic_messages decisions \
    responses responses_tools score tokenization tokenization_validation tool_literal tool_union; do
    file="$source/apps/server/$name.cpp"
    if [[ -f "$file" ]]; then
        sources+=("$file")
    fi
done

lmcache_api=()
if [[ -f "$source/apps/server/batch_cache.cpp" ]]; then
    sources+=("$source"/apps/server/{batch_cache,batch_service_cache}.cpp
              "$source"/src/runtime/cache/{lmcache_client,lmcache_prefix,prefix_payload}.cpp)
    lmcache_api=(-DSGLANG_TEST_LMCACHE)
fi

for name in qwen_chat_template qwen_chat_output qwen_tool_constraints qwen_serving_profile \
    qwen_tool_output; do
    file="$source/src/runtime/model/$name.cpp"
    if [[ -f "$file" ]]; then
        sources+=("$file")
    fi
done

if [[ -f "$source/apps/server/chat_media.cpp" ]]; then
    sources+=("$source/src/media_preprocess.cpp" "$source/src/media_resize.cpp"
             "$source/src/media/multimodal_prompt.cpp")
fi

libraries=()
if [[ "$build" == - ]]; then
    sources+=("$source"/src/tokenizer/*.cpp "$source/src/json.cpp" "$source/src/utf8.cpp")
else
    build=$(realpath -- "$build")
    libraries=("$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a")
fi

cuda_headers=()
if [[ -n ${CONDA_PREFIX:-} && ! -f "$source/include/sglang/runtime/model/serving_profile.h" ]]; then
    cuda_headers=(-isystem "$CONDA_PREFIX/targets/x86_64-linux/include")
fi

media_api=()
batch_header=$(<"$source/include/sglang/decoder_batch.h")
if [[ "$batch_header" == *admit_multimodal* ]]; then
    media_api=(-DSGLANG_TEST_BATCH_MEDIA)
fi
if [[ "$batch_header" == *release_retired_allocations* ]]; then
    media_api+=(-DSGLANG_TEST_RETIRED_ALLOCATIONS)
fi
if [[ $(<"$source/include/sglang/decoder.h") == *paired_prefill_threshold* ]]; then
    media_api+=(-DSGLANG_TEST_PAIRED_PREFILL_THRESHOLD)
fi
if [[ $(<"$source/include/sglang/decoder.h") == *backend_policy* ]]; then
    media_api+=(-DSGLANG_TEST_BACKEND_POLICY)
fi
vision_header="$source/include/sglang/server/vision_completion.h"
if [[ -f "$vision_header" && $(<"$vision_header") == *'generate(const Decoder&'* ]]; then
    media_api+=(-DSGLANG_TEST_SERIAL_MEDIA)
fi

"${CXX:-c++}" -std=c++17 -O2 -pthread \
    -Wall -Wextra -Wpedantic -Werror -ffunction-sections -fdata-sections \
    -I"$source/include" "${cuda_headers[@]}" "${media_api[@]}" "${lmcache_api[@]}" "${sources[@]}" \
    "$tests/chat_cpu_decoder.cpp" "${libraries[@]}" \
    -Wl,--wrap=_ZNSt6chrono3_V212steady_clock3nowEv \
    -Wl,--gc-sections,--print-gc-sections,--no-undefined,-Map,"$output.map" \
    -o "$output"
