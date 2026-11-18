#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sglang::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value, std::less<>>;

class Value {
public:
    Value() = default;
    Value(bool value) : data_(value) {}
    Value(int value) : data_(std::int64_t(value)) {}
    Value(std::int64_t value) : data_(value) {}
    Value(std::uint64_t value) : data_(value) {}
    Value(double value) : data_(value) {}
    Value(const char* value) : data_(std::string(value)) {}
    Value(std::string value) : data_(std::move(value)) {}
    Value(Array value) : data_(std::move(value)) {}
    Value(Object value) : data_(std::move(value)) {}

    const Object* object() const { return std::get_if<Object>(&data_); }
    const Array* array() const { return std::get_if<Array>(&data_); }
    std::optional<std::string_view> string() const;
    std::optional<std::uint64_t> unsigned_integer() const;
    std::optional<bool> boolean() const;

private:
    std::variant<std::monostate, bool, std::string, std::int64_t, std::uint64_t, double,
                 Array, Object> data_;
};

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Value parse(std::string_view input);
const Value* get(const Object& object, std::string_view key);
std::optional<std::string_view> get_string(const Object& object, std::string_view key);
std::optional<bool> get_boolean(const Object& object, std::string_view key);
void append_quoted(std::string& output, std::string_view value);

}
