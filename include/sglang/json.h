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
using OrderedObject = std::vector<std::pair<std::string, Value>>;

struct WideInteger {
    std::string digits;
};

enum class Syntax { strict, ordered, benchmark, checkpoint, completion };

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
    Value(OrderedObject value) : data_(std::move(value)) {}
    Value(WideInteger value) : data_(std::move(value)) {}

    Object* object() { return std::get_if<Object>(&data_); }
    const Object* object() const { return std::get_if<Object>(&data_); }
    const OrderedObject* ordered_object() const { return std::get_if<OrderedObject>(&data_); }
    const Array* array() const { return std::get_if<Array>(&data_); }
    const WideInteger* wide_integer() const { return std::get_if<WideInteger>(&data_); }
    bool is_null() const { return std::holds_alternative<std::monostate>(data_); }
    std::optional<std::string_view> string() const;
    std::optional<std::int64_t> signed_integer() const;
    std::optional<std::uint64_t> unsigned_integer() const;
    std::optional<double> number() const;
    std::optional<bool> boolean() const;

private:
    std::variant<std::monostate, bool, std::string, std::int64_t, std::uint64_t, double,
                 Array, Object, OrderedObject, WideInteger> data_;
};

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NumberRangeError : public ParseError {
public:
    using ParseError::ParseError;
};

Value parse(std::string_view input);
Value parse(std::string_view input, Syntax syntax);
const Value* get(const Object& object, std::string_view key);
const Value* get(const OrderedObject& object, std::string_view key);
std::optional<std::string_view> get_string(const Object& object, std::string_view key);
std::optional<bool> get_boolean(const Object& object, std::string_view key);
void append_quoted(std::string& output, std::string_view value);
void append_quoted(std::string& output, std::string_view value, bool short_control_escapes);

}
