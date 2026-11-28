set_xmakever("3.1.2")
set_project("sglang.cpp")
set_languages("cxx17")
set_policy("build.ccache", false)
add_rules("mode.release")

package("sglang-cutlass-headers")
    set_kind("library", {headeronly = true})
    local archive = os.getenv("SGLANG_CUTLASS_ARCHIVE")
    if archive and archive ~= "" then
        set_urls("file://" .. archive)
    else
        set_urls("https://codeload.github.com/NVIDIA/cutlass/tar.gz/" ..
                 "f8acf1aa78f96ff2f1ba7662fe495e5535c461bf")
    end
    add_versions("4.9.0dev", "773d01d7a901646dd9efff90c6bff183399e7330eefb03684ef0c7a984cb42f2")
    on_install(function (package)
        os.cp("include/*", package:installdir("include"))
        os.cp("tools/util/include/cutlass/util", package:installdir("include/cutlass"))
        os.cp("LICENSE.txt", package:installdir())
    end)
package_end()

add_requires("sglang-cutlass-headers 4.9.0dev", {system = false})

local prefix = os.getenv("CONDA_PREFIX")
local bindir = path.join(prefix, "bin")
local toolkit = path.join(prefix, "targets/x86_64-linux")
local host = os.getenv("CXX")

set_config("cuda", prefix, {force = true})

toolchain("host-pixi")
    set_kind("standalone")
    set_toolset("cc", "gcc@" .. os.getenv("CC"))
    set_toolset("cxx", "gxx@" .. host)
    set_toolset("ld", "gxx@" .. host)
    set_toolset("cu", path.join(bindir, "nvcc"))
    set_toolset("culd", path.join(bindir, "nvcc"))
    set_toolset("cu-ccbin", "gxx@" .. host)
toolchain_end()

target("toolchain-probe")
    set_kind("binary")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/bin")
    add_cugencodes("sm_100")
    add_cuflags("--dependency-target-name=", {force = true})
    add_includedirs(path.join(toolkit, "include"))
    add_linkdirs(path.join(toolkit, "lib"))
    add_rpathdirs(path.join(toolkit, "lib"))
    add_syslinks("cudart")
    add_files("tools/toolchain-probe.cpp", "tools/toolchain-probe.cu")

target("nvrtc-probe")
    set_kind("binary")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/bin")
    add_includedirs(path.join(toolkit, "include"))
    add_linkdirs(path.join(toolkit, "lib"), path.join(toolkit, "lib/stubs"))
    add_rpathdirs(path.join(toolkit, "lib"))
    add_syslinks("nvrtc", "cuda")
    add_files("tools/nvrtc-probe.cpp")

rule("sglang.cuda.device-lto")
    before_link(function (target, opt)
        import("core.base.option")
        import("core.project.depend")
        import("core.tool.linker")
        import("utils.progress")

        local objects = {}
        for _, batch in pairs(target:sourcebatches()) do
            if batch.sourcekind == "cu" then
                table.join2(objects, batch.objectfiles)
            end
        end
        local stem = target:basename() .. "_gpucode.cu"
        local output = target:objectfile(path.join("rules", "cuda", "devlink", stem))
        table.insert(target:objectfiles(), output)

        local arch = target:values("sglang.cuda.lto_arch")
        local link = linker.load("gpucode", "cu", {target = target})
        local flags = {"-m64", "-dlink", "-dlto",
                       "-ccbin=" .. target:tool("cu-ccbin"),
                       "-gencode=arch=compute_" .. arch .. ",code=sm_" .. arch}
        local dryrun = option.get("dry-run")
        depend.on_changed(function ()
            progress.show(opt.progress, "${color.build.target}devlinking.$(mode) %s",
                          path.filename(output))
            if option.get("verbose") then
                print(link:linkcmd(objects, output, {linkflags = flags, rawargs = true}))
            end
            if not dryrun then
                assert(link:link(objects, output, {linkflags = flags}))
            end
        end, {dependfile = target:dependfile(output), lastmtime = os.mtime(output),
              changed = target:is_rebuilt(), values = {link:program(), flags},
              files = objects, dryrun = dryrun})
    end)
rule_end()

local function add_cuda_device_lto(arch, sources, shared_sources, shared_flags)
    add_rules("sglang.cuda.device-lto")
    set_values("sglang.cuda.lto_arch", arch)
    local flags = {"-rdc=true", "-gencode=arch=compute_" .. arch .. ",code=lto_" .. arch}
    for _, source in ipairs(sources) do
        add_files(source, {force = {cuflags = flags}})
    end

    local alias_flags = table.join(flags, shared_flags)
    on_load(function (target)
        local directory = path.join(target:autogendir(), "cuda-link")
        directory = path.absolute(directory, os.projectdir())
        os.mkdir(directory)
        for _, file in ipairs(shared_sources) do
            local source = path.absolute(file, os.projectdir())
            local alias = path.join(directory, target:name() .. "_" .. path.filename(file))
            if os.islink(alias) and os.readlink(alias) ~= source then
                os.rm(alias)
            end
            if not os.islink(alias) then
                os.ln(source, alias)
            end
            target:add("files", alias, {force = {cuflags = alias_flags}})
        end
    end)
end

local gqa_device_sources = {
    "src/text/gqa/device_geometry.cu",
    "src/text/gqa/decode_layout.cu",
    "src/text/gqa/prefill_common.cu",
}

target("sglang_cuda")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    set_values("cuda.rdc", false)
    add_cugencodes("sm_100")
    add_cuflags("--dependency-target-name=", {force = true})
    add_includedirs("include", path.join(toolkit, "include"), {public = true})
    add_linkdirs(path.join(toolkit, "lib"), {public = true})
    add_rpathdirs(path.join(toolkit, "lib"), {interface = true})
    add_syslinks("cudart", {public = true})
    add_deps("sglang_prefill_fp8_sm100", "sglang_cuda_support")
    add_files("src/text_decode_fp8.cu")
    add_files("src/runtime/execution/tp_verify_graph.cu")
    add_files("src/text/draft_seed_fc.cu")
    add_files("src/text/bf16_head_projection.cu")
    add_files("src/text/target_fp8.cu")
    add_files("src/runtime/execution/verification_row_copy.cu")
    add_files("src/text_decode_bandwidth.cu")
    add_files("src/text/gdn_chunk.cu", "src/text/gdn_replay_ssm.cu")
    add_files("src/decoder_sampling.cu", "src/decoder_constraint.cu",
              "src/text/decoder_logprobs.cu")
    add_cuda_device_lto("100", {
        "src/mxfp8_gqa.cu",
        "src/mxfp8_gqa_prefill.cu",
        "src/text_attention_gdn.cu",
        "src/text_attention_gqa.cu",
        "src/text_normalization.cu",
        "src/text_ops.cu",
        "src/text/fp8_projection.cu",
        "src/text/gdn_prefill.cu",
        "src/text/rms_quantize.cu",
        "src/text/text_cuda_primitives.cu",
        "src/text/gqa/draft_attention.cu",
        "src/text/gqa/kv_append.cu",
        "src/text/gqa/ordered_prefill.cu",
        "src/text/gqa/paged_decode.cu",
        "src/text/gqa/target_split.cu",
    }, gqa_device_sources, {"-DSGLANG_GQA_DEVICE_ONLY"})

target("sglang_prefill_fp8_sm100")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    set_values("cuda.rdc", false)
    add_cugencodes("sm_100a")
    add_cuflags("--dependency-target-name=", {force = true})
    add_includedirs("include", path.join(toolkit, "include"))
    add_deps("sglang_cuda_support")
    add_files("src/prefill_fp8_sm100.cu", "src/text/gqa/geometry.cpp")
    add_files("src/text/draft_seed_fc_sm100.cu")
    add_files("src/text/decode_proj_host.cu")
    add_packages("sglang-cutlass-headers")
    add_files("src/text/prefill_down_sm100.cu", {cuflags = "--expt-relaxed-constexpr"})
    add_cuda_device_lto("100a", table.join(gqa_device_sources, {
        "src/text/decode_proj_sm100.cu",
        "src/text/decode_proj_fp8.cu",
        "src/text/decode_proj_bf16.cu",
        "src/text/draft_depth.cu",
        "src/text/gqa_decode_long.cu",
        "src/text/gqa_decode_split.cu",
        "src/text/mxfp8_gqa_prefill_tmem.cu",
        "src/text/prefill_gdn_gates_sm100.cu",
        "src/text/tc_vocab.cu",
        "src/text/gqa/tmem_ops.cu",
        "src/text/gqa/tmem_staged.cu",
        "src/text/gqa/tmem_staged_kernels.cu",
    }), {"src/text/text_cuda_primitives.cu"}, {"-DSGLANG_TEXT_DEVICE_ONLY"})

target("sglang_cuda_support")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", path.join(toolkit, "include"))
    add_files("src/runtime/backend/cuda_error.cpp", "src/runtime/backend/device_arena.cpp",
              "src/runtime/backend/text_cuda_host.cpp", "src/text/text_dimensions.cpp")

target("sglang_json")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", {public = true})
    add_files("src/json.cpp", "src/utf8.cpp")

target("sglang_tokenizer")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", {public = true})
    add_deps("sglang_json")
    add_sysincludedirs(path.join(prefix, "include"))
    add_linkdirs(path.join(prefix, "lib"), {public = true})
    add_links("icuuc", "icudata", {public = true})
    add_rpathdirs(path.join(prefix, "lib"), {interface = true})
    add_files("src/tokenizer/*.cpp")

target("sglang_checkpoint")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", {public = true})
    add_deps("sglang_json")
    add_files("src/checkpoint.cpp", "src/runtime/model/qwen_checkpoint.cpp")
    add_files("src/runtime/model/qwen_text_model.cpp")

target("sglang_vision")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    set_values("cuda.rdc", false)
    add_cugencodes("sm_100")
    add_cuflags("--dependency-target-name=", {force = true})
    add_includedirs("include", path.join(toolkit, "include"), {public = true})
    add_linkdirs(path.join(toolkit, "lib"), {public = true})
    add_rpathdirs(path.join(toolkit, "lib"), {interface = true})
    add_syslinks("cudart", {public = true})
    add_deps("sglang_checkpoint", "sglang_cuda_support")
    add_files("src/vision/*.cu")

target("sglang_decoder")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", {public = true})
    add_deps("sglang_checkpoint", "sglang_cuda", "sglang_prefill_fp8_sm100", "sglang_tokenizer")
    add_files("src/decoder.cpp", "src/decoder_forward.cpp",
              "src/decoder_batch.cpp", "src/decoder_prefix_cache.cpp",
              "src/decoder_mtp.cpp", "src/decoder_prefill.cpp", "src/decoder_verify.cpp",
              "src/decoder_constraint.cpp", "src/media/decoder_media.cpp",
              "src/media/multimodal_prompt.cpp")
    add_files("src/text/decoder_graph.cpp", "src/runtime/model/decoder_weights.cpp",
              "src/runtime/model/qwen_request.cpp", "src/runtime/backend/cuda_resources.cpp",
              "src/runtime/backend/retired_request_blocks.cpp",
              "src/runtime/backend/cuda_backend_policy.cpp",
              "src/runtime/execution/decoder_execution_policy.cpp",
              "src/runtime/execution/decoder_prefill_workspace.cpp",
              "src/runtime/cache/kv_residency_policy.cpp", "src/runtime/cache/prefix_payload.cpp")
    add_files("src/runtime/model/decoder_batch_lane.cpp",
              "src/runtime/model/qwen_weight_shard.cpp", "src/runtime/model/qwen_weight_upload.cpp",
              "src/runtime/cache/prefix_cache_layout.cpp",
              "src/runtime/cache/prefix_residency.cpp", "src/runtime/cache/prefix_slab.cpp",
              "src/runtime/execution/batch_admission_order.cpp",
              "src/runtime/execution/decoder_batch_state.cpp",
              "src/runtime/execution/decoder_forward_attention.cpp",
              "src/runtime/execution/decoder_forward_layers.cpp",
              "src/runtime/execution/decoder_options.cpp",
              "src/runtime/execution/decoder_prefill_plan.cpp",
              "src/runtime/execution/decoder_prefill_staged_gqa.cpp",
              "src/runtime/execution/decoder_prefill_tiles.cpp",
              "src/runtime/execution/decoder_verify_workspace.cpp",
              "src/text/decoder_batch_plan.cpp", "src/text/draft_projection.cpp",
              "src/text/gqa/verify_batch.cpp", "src/text/label_logprobs.cpp",
              "src/text/tc_vocab.cpp")

target("sglang-generate")
    set_kind("binary")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/bin")
    add_deps("sglang_decoder")
    add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
    add_files("apps/sglang-generate.cpp")

option("media-codec-prefix")
    set_default("")
    set_showmenu(true)
    set_description("Prefix with libpng and FFmpeg development libraries for inline chat media")
option_end()

local function add_media_codecs()
    local codecs = get_config("media-codec-prefix")
    if codecs and #codecs > 0 then
        add_defines("SGLANG_MEDIA_CODECS")
        add_files("apps/server/media_png.cpp", "apps/server/media_mp4.cpp")
        add_sysincludedirs(path.join(codecs, "include"))
        add_linkdirs(path.join(codecs, "lib"))
        add_rpathdirs(path.join(codecs, "lib"))
        add_links("png16", "avformat", "avcodec", "swscale", "avutil")
    end
end

target("sglang-server")
    set_kind("binary")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/bin")
    add_deps("sglang_decoder", "sglang_tokenizer", "sglang_media_preprocess", "sglang_vision")
    add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
    add_files("apps/sglang-server.cpp", "apps/server/batch_service.cpp",
              "apps/server/batch_cache.cpp",
              "apps/server/completions.cpp",
              "apps/server/flush_cache.cpp", "apps/server/http.cpp", "apps/server/loads.cpp",
              "src/runtime/server/single_device_batch_executor.cpp",
              "apps/server/prefix_controls.cpp",
              "apps/server/text_request.cpp", "apps/server/text_completion.cpp",
              "apps/server/chat_media.cpp", "apps/server/vision_completion.cpp")
    add_files("apps/server/anthropic_count.cpp", "apps/server/anthropic_messages.cpp",
              "apps/server/decisions.cpp",
              "apps/server/native_generate.cpp", "apps/server/native_sessions.cpp",
              "apps/server/ollama_discovery.cpp",
              "apps/server/ollama_generate.cpp",
              "apps/server/parse_function_call.cpp", "apps/server/responses.cpp",
              "apps/server/responses_tools.cpp", "apps/server/score.cpp",
              "apps/server/tokenization.cpp",
              "apps/server/tokenization_validation.cpp", "apps/server/tool_literal.cpp",
              "apps/server/tool_union.cpp")
    add_files("apps/server/detail/*.cpp",
              "src/runtime/server/batch_executor.cpp", "src/runtime/server/batch_restore_join.cpp",
              "src/runtime/server/cache_identity.cpp")
    add_files("apps/server/qwen/*.cpp")
    add_media_codecs()

option("text-tokenizer-check")
    set_default(false)
option_end()

if has_config("text-tokenizer-check") then
    target("text-tokenizer-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_tokenizer")
        add_files("tests/text/tokenizer_test.cpp")
    target_end()
end

option("mxfp8-gqa-check")
    set_default(false)
option_end()

if has_config("mxfp8-gqa-check") then
    target("mxfp8-gqa-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        set_values("cuda.rdc", false)
        set_values("cuda.build.devlink", false)
        add_cugencodes("sm_100")
        add_cuflags("--dependency-target-name=", {force = true})
        add_deps("sglang_cuda")
        add_syslinks("cudart")
        add_files("tests/mxfp8_gqa_check.cu", "tests/mxfp8_gqa_prefill_check.cu")
        add_files("tests/mxfp8_gqa_flash_check.cu", "tests/mxfp8_gqa_short_prefill_check.cu")
    target_end()

end

option("sampling-check")
    set_default(false)
option_end()

if has_config("sampling-check") then
    target("sampling-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_files("tests/text/sampling_check.cpp")
    target_end()
end

option("gdn-check")
    set_default(false)
option_end()

if has_config("gdn-check") then
    target("gdn-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        set_values("cuda.rdc", false)
        set_values("cuda.build.devlink", false)
        add_cugencodes("sm_100")
        add_cuflags("--dependency-target-name=", {force = true})
        add_deps("sglang_cuda")
        add_defines("SGLANG_TEST_GDN_TILED", "SGLANG_TEST_GDN_CHUNK")
        add_ldflags("-Wl,--wrap=cudaLaunchKernel", {force = true})
        add_files("tests/decoder/gdn_check.cu")
    target_end()

    target("gdn-routes")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder", "sglang_tokenizer")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_ldflags("-Wl,--wrap=cudaLaunchKernel", {force = true})
        add_files("tests/decoder/gdn_routes.cpp", "tests/prefix/prefix_fixture.cpp")
    target_end()
end

option("batch-graph-lanes")
    set_default(false)
option_end()

if has_config("batch-graph-lanes") then
    target("batch-graph-lanes")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_files("tests/batch/batch_graph_lanes.cpp")
    target_end()
end

option("decoder-batch-plan-check")
    set_default(false)
option_end()

if has_config("decoder-batch-plan-check") then
    target("decoder-batch-plan-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_includedirs("include")
        add_files("tests/decoder/decoder_batch_plan_check.cpp",
                  "src/runtime/execution/batch_admission_order.cpp",
                  "src/text/decoder_batch_plan.cpp")
    target_end()
end

option("constraint-check")
    set_default(false)
option_end()

if has_config("constraint-check") then
    target("constraint-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder", "sglang_tokenizer")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_files("tests/text/constraint_check.cpp")
        add_ldflags("-Wl,--wrap=_ZN6sglang7decoder25gather_constrained_logits" ..
                    "EPK13__nv_bfloat16PKiPS1_iP11CUstream_st", {force = true})
    target_end()
end

target("sglang_media_preprocess")
    set_kind("static")
    set_default(false)
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    add_includedirs("include", {public = true})
    add_cxxflags("-ffp-contract=off")
    add_files("src/media_preprocess.cpp", "src/media_resize.cpp")

option("media-preprocess-check")
    set_default(false)
option_end()

if has_config("media-preprocess-check") then
    target("media-preprocess-check")
        set_kind("binary")
        set_default(false)
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_media_preprocess")
        add_files("tests/media/preprocess_check.cpp")
    target_end()
end

option("long-decode-check")
    set_default(false)
option_end()

if has_config("long-decode-check") then
    target("gqa-split-fp64-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_cuda")
        add_files("tests/decoder/gqa_split_fp64_check.cpp")
    target_end()

    target("long-decode-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        set_values("cuda.rdc", false)
        set_values("cuda.build.devlink", false)
        add_cugencodes("sm_100")
        add_cuflags("--dependency-target-name=", {force = true})
        add_deps("sglang_cuda")
        add_ldflags("-Wl,--wrap=cudaLaunchKernel", {force = true})
        add_files("tests/decoder/long_decode_check.cu")
    target_end()

    target("long-decode-routes")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder", "sglang_tokenizer")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_ldflags("-Wl,--wrap=cudaLaunchKernel", {force = true})
        add_ldflags("-Wl,--wrap=_ZN6sglang7decoder15RequestProgress6selectERNS0_7RequestERNS0_" ..
                    "9WorkspaceERKNS_14DecoderOptionsERKNS0_19ConstraintWorkspaceEPKNS0_" ..
                    "18SamplingParametersEPvP11CUstream_stPKNS0_23AdjustedLogitsWorkspaceE",
                    {force = true})
        add_files("tests/decoder/long_decode_routes.cpp", "tests/prefix/prefix_fixture.cpp")
    target_end()
end

option("generated-logprobs-check")
    set_default(false)
option_end()

if has_config("generated-logprobs-check") then
    target("generated-logprobs-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_decoder")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_files("tests/text/generated_logprobs_check.cpp")
    target_end()
end
local nccl_wheel = (os.dirs(path.join(prefix, "lib/python*/site-packages/nvidia/nccl")) or {})[1]
if not nccl_wheel then
    raise("pixi env is missing the nvidia/nccl wheel " ..
          "(expected $CONDA_PREFIX/lib/python*/site-packages/nvidia/nccl)")
end

target("sglang_decoder")
    add_sysincludedirs(path.join(nccl_wheel, "include"))
    add_ldflags(path.join(nccl_wheel, "lib/libnccl.so.2"), {force = true, public = true})
    add_ldflags("-Wl,-rpath," .. path.join(nccl_wheel, "lib"), {force = true, public = true})
    add_files("src/runtime/backend/tensor_parallel.cpp")
    add_files("src/runtime/cache/lmcache_group.cpp", "src/runtime/cache/lmcache_client.cpp",
              "src/runtime/cache/lmcache_prefix.cpp")
    add_files("src/decoder_tp.cpp")
    add_files("src/runtime/execution/decoder_forward_tp.cpp")
    add_files("src/runtime/execution/tp_verify_graph.cpp")
    add_files("src/runtime/execution/peer_draft_repair.cpp")
    add_files("src/runtime/execution/decoder_batch_prefix.cpp")
target_end()

option("tp2-collective-check")
    set_default(false)
option_end()

if has_config("tp2-collective-check") and os.isfile("tests/decoder/tp2/collective_check.cpp") then
    target("tp2-collective-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_sysincludedirs(path.join(nccl_wheel, "include"))
        add_deps("sglang_decoder")
        add_rpathdirs(path.join(prefix, "lib"), path.join(toolkit, "lib"))
        add_ldflags("-Wl,--wrap=cuMulticastGetGranularity", {force = true})
        add_files("tests/decoder/tp2/collective_check.cpp")
    target_end()
end

option("tp2-verification-row-copy-check")
    set_default(false)
option_end()

if has_config("tp2-verification-row-copy-check") then
    target("tp2-verification-row-copy-check")
        set_kind("binary")
        set_toolchains("host-pixi")
        set_targetdir("$(builddir)/bin")
        add_deps("sglang_cuda")
        add_files("tests/decoder/tp2/verification_row_copy_check.cpp")
    target_end()
end

target("sglang_mnnvl_ar")
    set_kind("static")
    set_toolchains("host-pixi")
    set_targetdir("$(builddir)/lib")
    set_values("cuda.rdc", false)
    add_cugencodes("sm_100a")
    add_cuflags("--dependency-target-name=", {force = true})
    add_cuflags("--expt-relaxed-constexpr", {force = true})
    add_includedirs("include", path.join(toolkit, "include"))
    add_syslinks("cuda", {public = true})
    add_deps("sglang_cuda_support")
    add_cuda_device_lto("100a", {"src/runtime/backend/mnnvl_ar.cu"},
                        {"src/text/rms_quantize.cu", "src/text/text_cuda_primitives.cu"},
                        {"-DSGLANG_TEXT_DEVICE_ONLY"})
target_end()

target("sglang_decoder")
    add_deps("sglang_mnnvl_ar")
    add_defines("SGLANG_MNNVL_AR=1")
target_end()
