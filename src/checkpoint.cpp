#include "sglang/checkpoint.h"

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

#include <llvm/Support/JSON.h>

namespace sglang {
namespace {

namespace json = llvm::json;
using Path = std::filesystem::path;

constexpr std::size_t max_metadata_bytes = 8 * 1024 * 1024;
constexpr std::size_t max_json_depth = 32;
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

bool is_decimal_digit(char character) {
    return character >= '0' && character <= '9';
}

bool consume_digits(llvm::StringRef& input) {
    const auto size = input.size();
    while (!input.empty() && is_decimal_digit(input.front())) {
        input = input.drop_front();
    }
    return input.size() != size;
}

bool consume_json_number(llvm::StringRef& input) {
    input.consume_front("-");
    if (!input.consume_front("0") && !consume_digits(input)) {
        return false;
    }
    if (input.consume_front(".") && !consume_digits(input)) {
        return false;
    }
    if (input.consume_front("e") || input.consume_front("E")) {
        if (!input.consume_front("+")) {
            input.consume_front("-");
        }
        if (!consume_digits(input)) {
            return false;
        }
    }
    return input.empty() || llvm::StringRef(",]} \t\r\n").contains(input.front());
}

std::size_t count_json_keys(llvm::StringRef source, const Path& path) {
    std::size_t depth = 0;
    std::size_t keys = 0;
    bool in_string = false;
    bool escaped = false;
    while (!source.empty()) {
        const char character = source.front();
        if (!in_string && (is_decimal_digit(character) || character == '-' ||
                           character == '+' || character == '.')) {
            if (!consume_json_number(source)) {
                invalid(path, "invalid JSON number");
            }
            continue;
        }
        source = source.drop_front();
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                in_string = false;
            }
        } else if (character == '"') {
            in_string = true;
        } else if (character == '{' || character == '[') {
            if (++depth > max_json_depth) {
                invalid(path, "JSON nesting exceeds the checkpoint metadata limit");
            }
        } else if (character == '}' || character == ']') {
            if (depth == 0) {
                invalid(path, "unbalanced JSON delimiters");
            }
            --depth;
        } else if (character == ':') {
            ++keys;
        }
    }
    return keys;
}

std::size_t count_object_keys(const json::Value& value) {
    if (const auto* object = value.getAsObject()) {
        std::size_t keys = object->size();
        for (const auto& field : *object) {
            keys += count_object_keys(field.second);
        }
        return keys;
    }
    std::size_t keys = 0;
    if (const auto* array = value.getAsArray()) {
        for (const auto& entry : *array) {
            keys += count_object_keys(entry);
        }
    }
    return keys;
}

json::Object parse_object(const Path& path, const std::byte* data, std::size_t size) {
    const llvm::StringRef source(reinterpret_cast<const char*>(data), size);
    const std::size_t keys = count_json_keys(source, path);
    auto value = json::parse(source);
    if (!value) {
        invalid(path, llvm::toString(value.takeError()));
    }
    auto* object = value->getAsObject();
    if (!object) {
        invalid(path, "expected a JSON object");
    }
    if (count_object_keys(*value) != keys) {
        invalid(path, "duplicate JSON object key");
    }
    return std::move(*object);
}

json::Object read_json_file(const Path& path) {
    const Mapping file(path, max_metadata_bytes);
    return parse_object(path, file.data, file.size);
}

const json::Object& require_object(const json::Object& parent, const char* name, const Path& path) {
    const auto* object = parent.getObject(name);
    if (!object) {
        invalid(path, std::string("expected a JSON object at ") + name);
    }
    return *object;
}

void check_fields(const json::Object& actual, std::initializer_list<json::Object::KV> expected,
                  const Path& path, const std::string& prefix) {
    for (const auto& field : expected) {
        const auto* value = actual.get(field.K);
        if (!value || *value != field.V) {
            invalid(path, prefix + field.K.str() + " differs from the pinned text model");
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

bool valid_shard_name(llvm::StringRef name) {
    return name.ends_with(".safetensors") && name.size() > sizeof(".safetensors") - 1 &&
           name.find_first_not_of(
               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") ==
               llvm::StringRef::npos;
}

ShardIndex read_index(const Path& path) {
    const auto index = read_json_file(path);
    const auto& weights = require_object(index, "weight_map", path);
    if (weights.empty()) {
        invalid(path, "empty weight_map");
    }
    ShardIndex shards;
    for (const auto& entry : weights) {
        const llvm::StringRef name = entry.first;
        const auto shard = entry.second.getAsString();
        if (name.empty() || name.contains('\0') || name == "__metadata__" ||
            !shard || !valid_shard_name(*shard)) {
            invalid(path, "invalid weight_map entry for " + name.str());
        }
        shards[shard->str()].insert(name.str());
    }
    return shards;
}

struct TensorType {
    TensorDtype dtype;
    std::uint64_t width;
};

TensorType tensor_type(const json::Object& spec, const Path& path, llvm::StringRef name) {
    const auto dtype = spec.getString("dtype");
    if (dtype == "BF16") {
        return {TensorDtype::BF16, 2};
    }
    if (dtype == "F8_E4M3") {
        return {TensorDtype::F8_E4M3, 1};
    }
    invalid(path, "unsupported dtype for " + name.str());
}

std::vector<std::uint64_t> tensor_shape(const json::Object& spec, const Path& path,
                                       llvm::StringRef name) {
    const auto* dimensions = spec.getArray("shape");
    if (!dimensions) {
        invalid(path, "missing or invalid shape for " + name.str());
    }
    std::vector<std::uint64_t> shape;
    shape.reserve(dimensions->size());
    for (const auto& entry : *dimensions) {
        const auto dimension = entry.getAsUINT64();
        if (!dimension) {
            invalid(path, "invalid dimension for " + name.str());
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
                       const Path& path, llvm::StringRef name) {
    const auto* offsets = spec.getArray("data_offsets");
    if (!offsets || offsets->size() != 2) {
        invalid(path, "invalid data_offsets for " + name.str());
    }
    const auto begin = (*offsets)[0].getAsUINT64();
    const auto end = (*offsets)[1].getAsUINT64();
    if (!begin || !end || *end < *begin || *end > payload_size) {
        invalid(path, "data_offsets out of bounds for " + name.str());
    }
    return {*begin, *end};
}

void check_tensor_size(const std::vector<std::uint64_t>& shape, std::uint64_t width,
                       std::uint64_t available, const Path& path, llvm::StringRef name) {
    std::uint64_t bytes = 0;
    if (std::find(shape.begin(), shape.end(), 0) == shape.end()) {
        bytes = width;
        for (const auto dimension : shape) {
            if (dimension > available / bytes) {
                invalid(path, "shape and dtype exceed the byte range for " + name.str());
            }
            bytes *= dimension;
        }
    }
    if (bytes != available) {
        invalid(path, "shape and dtype do not match the byte range for " + name.str());
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
    const auto* metadata = value.getAsObject();
    if (!metadata) {
        invalid(path, "invalid __metadata__ object");
    }
    for (const auto& entry : *metadata) {
        if (!entry.second.getAsString()) {
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

    for (const auto& entry : header) {
        const llvm::StringRef name = entry.first;
        if (name == "__metadata__") {
            validate_shard_metadata(entry.second, path);
            continue;
        }
        const auto expected = missing.find(std::string_view(name.data(), name.size()));
        if (expected == missing.end()) {
            invalid(path, "tensor is not indexed to this shard: " + name.str());
        }
        const auto* spec = entry.second.getAsObject();
        if (!spec) {
            invalid(path, "invalid tensor metadata for " + name.str());
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
            invalid(path, "duplicate tensor name: " + name.str());
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
    std::list<Mapping> shards;
    TensorMap tensors;
};

Checkpoint::Checkpoint(const std::filesystem::path& directory) {
    validate_text_config(directory / "config.json");
    auto index = read_index(directory / "model.safetensors.index.json");
    auto storage = std::make_unique<Storage>();
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

}
