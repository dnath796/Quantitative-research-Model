// Minimal self-contained JSON reader for the flat golden.json schema
// ({"cases": [{"name": str, "inputs": {...}, "expect": {...}, "tol": num}]}).
// Supports objects, arrays, strings (common escapes), numbers, booleans and
// null — enough for any flat schema; \uXXXX escapes are rejected. Test-only.

#ifndef DPE_TESTS_MINI_JSON_HPP
#define DPE_TESTS_MINI_JSON_HPP

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace mini_json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
 public:
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> v;

  bool is_number() const { return std::holds_alternative<double>(v); }
  bool is_string() const { return std::holds_alternative<std::string>(v); }

  double as_number() const {
    if (!is_number()) throw std::runtime_error("JSON value is not a number");
    return std::get<double>(v);
  }
  const std::string& as_string() const {
    if (!is_string()) throw std::runtime_error("JSON value is not a string");
    return std::get<std::string>(v);
  }
  const Array& as_array() const {
    if (!std::holds_alternative<Array>(v)) throw std::runtime_error("not an array");
    return std::get<Array>(v);
  }
  const Object& as_object() const {
    if (!std::holds_alternative<Object>(v)) throw std::runtime_error("not an object");
    return std::get<Object>(v);
  }
  bool contains(const std::string& key) const {
    return as_object().count(key) != 0;
  }
  const Value& at(const std::string& key) const {
    const Object& obj = as_object();
    auto it = obj.find(key);
    if (it == obj.end()) throw std::runtime_error("missing JSON key: " + key);
    return it->second;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value parse() {
    Value v = parse_value();
    skip_ws();
    if (pos_ != text_.size()) throw std::runtime_error("trailing JSON content");
    return v;
  }

 private:
  const std::string& text_;
  std::size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) +
                             ": " + what);
  }

  void skip_ws() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
      ++pos_;
    }
  }

  char peek() {
    if (pos_ >= text_.size()) fail("unexpected end of input");
    return text_[pos_];
  }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool consume_literal(const char* lit) {
    std::size_t len = 0;
    while (lit[len] != '\0') ++len;
    if (text_.compare(pos_, len, lit) == 0) {
      pos_ += len;
      return true;
    }
    return false;
  }

  Value parse_value() {
    skip_ws();
    const char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return Value{parse_string()};
    if (consume_literal("true")) return Value{true};
    if (consume_literal("false")) return Value{false};
    if (consume_literal("null")) return Value{nullptr};
    return parse_number();
  }

  Value parse_object() {
    expect('{');
    Object obj;
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return Value{std::move(obj)};
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      obj.emplace(std::move(key), parse_value());
      skip_ws();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect('}');
      return Value{std::move(obj)};
    }
  }

  Value parse_array() {
    expect('[');
    Array arr;
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return Value{std::move(arr)};
    }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      if (peek() == ',') {
        ++pos_;
        continue;
      }
      expect(']');
      return Value{std::move(arr)};
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const char c = text_[pos_++];
      if (c == '"') return out;
      if (c == '\\') {
        if (pos_ >= text_.size()) fail("unterminated escape");
        const char e = text_[pos_++];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          default: fail("unsupported escape sequence");
        }
      } else {
        out += c;
      }
    }
  }

  Value parse_number() {
    const std::size_t start = pos_;
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' ||
          c == 'E') {
        ++pos_;
      } else {
        break;
      }
    }
    if (pos_ == start) fail("invalid token");
    const std::string token = text_.substr(start, pos_ - start);
    char* end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == nullptr || *end != '\0') fail("invalid number: " + token);
    return Value{value};
  }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace mini_json

#endif  // DPE_TESTS_MINI_JSON_HPP
