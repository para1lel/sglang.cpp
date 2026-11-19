#include "sglang/checkpoint.h"

#include "sglang/json.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sglang {
namespace {

using Path = std::filesystem::path;

constexpr std::size_t max_metadata_bytes = 8 * 1024 * 1024;
constexpr std::size_t max_mapping_bytes = std::numeric_limits<std::ptrdiff_t>::max();
constexpr std::size_t shard_prefix_bytes = 8;

[[noreturn]] void invalid(const Path& path, const std::string& reason) {
    throw std::runtime_error(path.string() + ": " + reason);
}

[[noreturn]] void file_error(const Path& path, const char* operation) {
    const int error = errno;
    throw std::system_error(error, std::generic_category(), operation + (" " + path.string()));
}

class Mapping {
public:
    explicit Mapping(const Path& path, std::size_t limit = max_mapping_bytes) {
        struct FileDescriptor {
            int handle;
            ~FileDescriptor() {
                if (handle >= 0) {
                    close(handle);
                }
            }
        };
        const FileDescriptor file{open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)};
        if (file.handle < 0) {
            file_error(path, "open");
        }
        struct stat status {};
        if (fstat(file.handle, &status) < 0) {
            file_error(path, "fstat");
        }
        if (!S_ISREG(status.st_mode) || status.st_size <= 0 ||
            static_cast<std::uint64_t>(status.st_size) > limit) {
            invalid(path, "invalid checkpoint file type or size");
        }
        size = static_cast<std::size_t>(status.st_size);
        void* address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file.handle, 0);
        if (address == MAP_FAILED) {
            file_error(path, "mmap");
        }
        data = static_cast<const std::byte*>(address);
    }

    ~Mapping() { munmap(const_cast<std::byte*>(data), size); }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    const std::byte* data;
    std::size_t size;
};

json::Object parse_object(const Path& path, const std::byte* data, std::size_t size) {
    try {
        auto value = json::parse(
            {reinterpret_cast<const char*>(data), size}, json::Syntax::checkpoint);
        if (auto* object = value.object()) {
            return std::move(*object);
        }
    } catch (const json::ParseError& error) {
        invalid(path, error.what());
    }
    invalid(path, "expected a JSON object");
}

json::Object read_json_file(const Path& path) {
    const Mapping file(path, max_metadata_bytes);
    return parse_object(path, file.data, file.size);
}

const json::Object& require_object(const json::Object& parent, const char* name, const Path& path) {
    const auto* value = json::get(parent, name);
    const auto* object = value ? value->object() : nullptr;
    if (!object) {
        invalid(path, std::string("expected a JSON object at ") + name);
    }
    return *object;
}

bool matches_config(const json::Value& actual, const json::Value& expected) {
    if (const auto* expected_array = expected.array()) {
        const auto* actual_array = actual.array();
        return actual_array && std::equal(actual_array->begin(), actual_array->end(),
                                         expected_array->begin(), expected_array->end(),
                                         matches_config);
    }
    if (const auto text = expected.string()) {
        return actual.string() == text;
    }
    if (const auto flag = expected.boolean()) {
        return actual.boolean() == flag;
    }
    const auto number = expected.number();
    return number && actual.number() == number;
}

void check_fields(const json::Object& actual,
                  std::initializer_list<std::pair<const char*, json::Value>> expected,
                  const Path& path, const std::string& prefix, const char* model = "text") {
    for (const auto& [name, expected_value] : expected) {
        const auto* value = json::get(actual, name);
        if (!value || !matches_config(*value, expected_value)) {
            invalid(path, prefix + name + " differs from the pinned " + model + " model");
        }
    }
}

void validate_text_config(const Path& path) {
    const auto config = read_json_file(path);
    check_fields(config, {
        {"architectures", json::Array{"Qwen3_5ForConditionalGeneration"}},
        {"model_type", "qwen3_5"},
        {"tie_word_embeddings", false},
    }, path, "");

    const auto& text = require_object(config, "text_config", path);
    json::Array layers;
    for (int layer = 0; layer < 64; ++layer) {
        layers.emplace_back(layer % 4 == 3 ? "full_attention" : "linear_attention");
    }
    check_fields(text, {
        {"model_type", "qwen3_5_text"},
        {"dtype", "bfloat16"},
        {"hidden_act", "silu"},
        {"hidden_size", 5120},
        {"intermediate_size", 17408},
        {"vocab_size", 248320},
        {"tie_word_embeddings", false},
        {"num_hidden_layers", 64},
        {"full_attention_interval", 4},
        {"layer_types", std::move(layers)},
        {"rms_norm_eps", 1e-6},
        {"num_attention_heads", 24},
        {"num_key_value_heads", 4},
        {"head_dim", 256},
        {"attention_bias", false},
        {"attention_dropout", 0.0},
        {"attn_output_gate", true},
        {"linear_conv_kernel_dim", 4},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_num_key_heads", 16},
        {"linear_num_value_heads", 48},
        {"output_gate_type", "swish"},
        {"mamba_ssm_dtype", "float32"},
        {"max_position_embeddings", 262144},
        {"partial_rotary_factor", 0.25},
    }, path, "text_config.");

    const auto& rope = require_object(text, "rope_parameters", path);
    check_fields(rope, {
        {"rope_type", "default"},
        {"rope_theta", 10000000},
        {"partial_rotary_factor", 0.25},
        {"mrope_interleaved", true},
        {"mrope_section", json::Array{11, 11, 10}},
    }, path, "text_config.rope_parameters.");

    const auto& quantization = require_object(config, "quantization_config", path);
    check_fields(quantization, {
        {"quant_method", "fp8"},
        {"fmt", "e4m3"},
        {"activation_scheme", "dynamic"},
        {"weight_block_size", json::Array{128, 128}},
    }, path, "quantization_config.");
}

using TensorNames = std::set<std::string, std::less<>>;
using ShardIndex = std::map<std::string, TensorNames>;
using TensorMap = std::map<std::string, HostTensorView, std::less<>>;

bool valid_shard_name(std::string_view name) {
    constexpr std::string_view extension = ".safetensors";
    return name.size() > extension.size() &&
           name.substr(name.size() - extension.size()) == extension &&
           name.find_first_not_of(
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") ==
               std::string_view::npos;
}

ShardIndex read_index(const Path& path) {
    const auto index = read_json_file(path);
    const auto& weights = require_object(index, "weight_map", path);
    if (weights.empty()) {
        invalid(path, "empty weight_map");
    }
    ShardIndex shards;
    for (const auto& [name, value] : weights) {
        const auto shard = value.string();
        if (name.empty() || name.find('\0') != std::string::npos || name == "__metadata__" ||
            !shard || !valid_shard_name(*shard)) {
            invalid(path, "invalid weight_map entry for " + name);
        }
        shards[std::string(*shard)].insert(name);
    }
    return shards;
}

struct TensorType {
    TensorDtype dtype;
    std::uint64_t width;
};

TensorType tensor_type(const json::Object& spec, const Path& path, const std::string& name) {
    const auto dtype = json::get_string(spec, "dtype");
    if (dtype == "BF16") {
        return {TensorDtype::BF16, 2};
    }
    if (dtype == "F8_E4M3") {
        return {TensorDtype::F8_E4M3, 1};
    }
    invalid(path, "unsupported dtype for " + name);
}

std::vector<std::uint64_t> tensor_shape(const json::Object& spec, const Path& path,
                                       const std::string& name) {
    const auto* field = json::get(spec, "shape");
    const auto* dimensions = field ? field->array() : nullptr;
    if (!dimensions) {
        invalid(path, "missing or invalid shape for " + name);
    }
    std::vector<std::uint64_t> shape;
    shape.reserve(dimensions->size());
    for (const auto& entry : *dimensions) {
        const auto dimension = entry.unsigned_integer();
        if (!dimension) {
            invalid(path, "invalid dimension for " + name);
        }
        shape.push_back(*dimension);
    }
    return shape;
}

struct ByteRange {
    std::uint64_t begin;
    std::uint64_t end;
};

ByteRange tensor_range(const json::Object& spec, std::uint64_t payload_size,
                       const Path& path, const std::string& name) {
    const auto* field = json::get(spec, "data_offsets");
    const auto* offsets = field ? field->array() : nullptr;
    if (!offsets || offsets->size() != 2) {
        invalid(path, "invalid data_offsets for " + name);
    }
    const auto begin = (*offsets)[0].unsigned_integer();
    const auto end = (*offsets)[1].unsigned_integer();
    if (!begin || !end || *end < *begin || *end > payload_size) {
        invalid(path, "data_offsets out of bounds for " + name);
    }
    return {*begin, *end};
}

void check_tensor_size(const std::vector<std::uint64_t>& shape, std::uint64_t width,
                       std::uint64_t available, const Path& path, const std::string& name) {
    std::uint64_t bytes = 0;
    if (std::find(shape.begin(), shape.end(), 0) == shape.end()) {
        bytes = width;
        for (const auto dimension : shape) {
            if (dimension > available / bytes) {
                invalid(path, "shape and dtype exceed the byte range for " + name);
            }
            bytes *= dimension;
        }
    }
    if (bytes != available) {
        invalid(path, "shape and dtype do not match the byte range for " + name);
    }
}

std::size_t shard_header_size(const Mapping& file, const Path& path) {
    if (file.size <= shard_prefix_bytes) {
        invalid(path, "missing safetensors header");
    }
    std::uint64_t size = 0;
    for (std::size_t byte = 0; byte < shard_prefix_bytes; ++byte) {
        size |= std::to_integer<std::uint64_t>(file.data[byte]) << (8 * byte);
    }
    if (size == 0 || size > max_metadata_bytes || size > file.size - shard_prefix_bytes ||
        file.data[shard_prefix_bytes] != std::byte{'{'}) {
        invalid(path, "invalid safetensors header size or prefix");
    }
    return static_cast<std::size_t>(size);
}

void validate_shard_metadata(const json::Value& value, const Path& path) {
    const auto* metadata = value.object();
    if (!metadata) {
        invalid(path, "invalid __metadata__ object");
    }
    for (const auto& entry : *metadata) {
        if (!entry.second.string()) {
            invalid(path, "non-string value in __metadata__");
        }
    }
}

void validate_coverage(std::vector<ByteRange>& ranges, std::uint64_t payload_size,
                       const Path& path) {
    std::sort(ranges.begin(), ranges.end(), [](const ByteRange& left, const ByteRange& right) {
        return left.begin != right.begin ? left.begin < right.begin : left.end < right.end;
    });
    std::uint64_t cursor = 0;
    for (const auto& range : ranges) {
        if (range.begin != cursor) {
            invalid(path, "tensor byte ranges overlap or leave a gap");
        }
        cursor = range.end;
    }
    if (cursor != payload_size) {
        invalid(path, "tensor byte ranges do not reach the end of the payload");
    }
}

void load_shard(const Path& path, const Mapping& file, TensorNames& missing, TensorMap& tensors) {
    const auto header_size = shard_header_size(file, path);
    const auto payload_offset = shard_prefix_bytes + header_size;
    const auto payload_size = file.size - payload_offset;
    const auto header = parse_object(path, file.data + shard_prefix_bytes, header_size);
    std::vector<ByteRange> ranges;
    ranges.reserve(missing.size());

    for (const auto& [name, value] : header) {
        if (name == "__metadata__") {
            validate_shard_metadata(value, path);
            continue;
        }
        const auto expected = missing.find(name);
        if (expected == missing.end()) {
            invalid(path, "tensor is not indexed to this shard: " + name);
        }
        const auto* spec = value.object();
        if (!spec) {
            invalid(path, "invalid tensor metadata for " + name);
        }
        const auto type = tensor_type(*spec, path, name);
        auto shape = tensor_shape(*spec, path, name);
        const auto range = tensor_range(*spec, payload_size, path, name);
        const auto bytes = range.end - range.begin;
        check_tensor_size(shape, type.width, bytes, path, name);
        ranges.push_back(range);
        const auto* data = file.data + payload_offset + range.begin;
        if (!tensors.emplace(*expected, HostTensorView{
                type.dtype, std::move(shape), data, static_cast<std::size_t>(bytes)
            }).second) {
            invalid(path, "duplicate tensor name: " + name);
        }
        missing.erase(expected);
    }
    if (!missing.empty()) {
        invalid(path, "indexed tensor missing from shard: " + *missing.begin());
    }
    validate_coverage(ranges, payload_size, path);
}

}

struct Checkpoint::Storage {
    Path config;
    std::list<Mapping> shards;
    TensorMap tensors;
};

Checkpoint::Checkpoint(const std::filesystem::path& directory) {
    validate_text_config(directory / "config.json");
    auto index = read_index(directory / "model.safetensors.index.json");
    auto storage = std::make_unique<Storage>();
    storage->config = std::filesystem::absolute(directory / "config.json");
    for (auto& [shard, missing] : index) {
        const auto path = directory / shard;
        const auto& file = storage->shards.emplace_back(path);
        load_shard(path, file, missing, storage->tensors);
    }
    storage_ = std::move(storage);
}

Checkpoint::~Checkpoint() = default;

const HostTensorView& Checkpoint::tensor(std::string_view name) const {
    const auto tensor = storage_->tensors.find(name);
    if (tensor == storage_->tensors.end()) {
        throw std::runtime_error("missing checkpoint tensor: " + std::string(name));
    }
    return tensor->second;
}

const HostTensorView& Checkpoint::tensor(std::string_view name, TensorDtype dtype,
                                         std::initializer_list<std::uint64_t> shape) const {
    const auto& view = tensor(name);
    if (view.dtype != dtype || !std::equal(shape.begin(), shape.end(), view.shape.begin(),
                                          view.shape.end())) {
        throw std::runtime_error("unexpected checkpoint tensor dtype or shape: " +
                                 std::string(name));
    }
    return view;
}

void Checkpoint::validate_vision() const {
    const auto& path = storage_->config;
    const auto config = read_json_file(path);
    const auto& vision = require_object(config, "vision_config", path);
    check_fields(vision, {
        {"model_type", "qwen3_5"},
        {"depth", 27},
        {"hidden_size", 1152},
        {"intermediate_size", 4304},
        {"out_hidden_size", 5120},
        {"num_heads", 16},
        {"num_position_embeddings", 2304},
        {"in_channels", 3},
        {"patch_size", 16},
        {"temporal_patch_size", 2},
        {"spatial_merge_size", 2},
        {"hidden_act", "gelu_pytorch_tanh"},
        {"deepstack_visual_indexes", json::Array{}},
    }, path, "vision_config.", "vision");
    constexpr std::string_view prefix = "model.visual.";
    auto tensor = storage_->tensors.lower_bound(prefix);
    int count = 0;
    for (; tensor != storage_->tensors.end() &&
           std::string_view(tensor->first).substr(0, prefix.size()) == prefix; ++tensor) {
        ++count;
    }
    if (count != 333) {
        invalid(path, "expected 333 indexed vision tensors");
    }
}

}
