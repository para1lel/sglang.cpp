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

enum class Syntax {
    strict, ordered, benchmark, checkpoint, completion, native_generate, last_key_wins,
    ordered_last_key_wins, tool_history, chat_tokenization, python_request, http_body
};

class Value {
public:
    Value() noexcept;
    Value(bool value);
    Value(int value);
    Value(std::int64_t value);
    Value(std::uint64_t value);
    Value(double value);
    Value(const char* value);
    Value(std::string value);
    Value(Array value);
    Value(Object value);
    Value(OrderedObject value);
    Value(WideInteger value);

    Object* object();
    const Object* object() const;
    OrderedObject* ordered_object();
    const OrderedObject* ordered_object() const;
    Array* array();
    const Array* array() const;
    const WideInteger* wide_integer() const;
    bool is_null() const;
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
const Value* get_ordered(const Value& value, std::string_view key);
std::optional<std::string_view> get_string(const Object& object, std::string_view key);
std::optional<bool> get_boolean(const Object& object, std::string_view key);
void append_quoted(std::string& output, std::string_view value);
void append_quoted(std::string& output, std::string_view value, bool short_control_escapes);

}
