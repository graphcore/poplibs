// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_OptionParsing_hpp
#define poplibs_support_OptionParsing_hpp

#include <functional>
#include <initializer_list>
#include <map>
#include <poplar/StringRef.hpp>
#include <poplar/exceptions.hpp>
#include <sstream>
#include <string>

namespace poplibs {

// OptionSpec/OptionHandler used to build up a specification of what
// options and their values should be, and to translate the value strings
// to real values.
class OptionHandler {
  std::function<void(poplar::StringRef)> valueHandler;

public:
  template <typename T>
  OptionHandler(T &&valueHandler)
      : valueHandler(std::forward<T>(valueHandler)) {}
  void parseValue(poplar::StringRef value) const { valueHandler(value); }

  // Utility functions to help build a spec
  template <typename T>
  static inline std::string
  describeEnumValues(const std::map<std::string, T> &valueMap) {
    std::stringstream s;
    s << "[";
    if (!valueMap.empty()) {
      auto it = valueMap.begin();
      s << "'" << it->first << "'";
      while (++it != valueMap.end()) {
        s << ", '" << it->first << "'";
      }
    }
    s << "]";
    return s.str();
  }

  template <typename T, typename ValueMapT = std::map<std::string, T>>
  static inline OptionHandler createWithEnum(T &output, ValueMapT &&valueMap) {

    // Allow forwarding of rvalues for valueMap but we always want
    // to hold a copy for safety.
    return OptionHandler{[map = std::forward<ValueMapT>(valueMap),
                          &output](poplar::StringRef value) {
      auto it = map.find(value);
      if (it == map.end()) {
        throw poplar::invalid_option("Not one of the values: " +
                                     describeEnumValues(map));
      }
      output = it->second;
    }};
  }

  template <typename T>
  static inline OptionHandler createWithInteger(T &output) {
    return OptionHandler{[&output](poplar::StringRef value) {
      auto stdStr = value.cloneAsString();
      int result;
      std::istringstream iss(stdStr);
      if (stdStr.find("0x") != std::string::npos) {
        iss >> std::hex >> result;
      } else {
        iss >> std::dec >> result;
      }
      if (iss.fail() || !iss.eof()) {
        throw poplar::invalid_option("Not a valid integer");
      }
      output = result;
    }};
  }

  template <typename T> static inline OptionHandler createWithBool(T &output) {
    static std::map<std::string, T> boolMap{{"true", static_cast<T>(true)},
                                            {"false", static_cast<T>(false)}};
    return createWithEnum(output, boolMap);
  }

  template <typename T>
  static inline OptionHandler createWithDouble(T &output) {
    return OptionHandler{[&output](poplar::StringRef value) {
      std::stringstream s(value);
      double val;
      s >> val;
      if (s.fail()) {
        throw poplar::invalid_option("Not a floating point number");
      }
      output = val;
    }};
  }

  static inline OptionHandler createWithString(std::string &output) {
    return OptionHandler{
        [&output](poplar::StringRef value) { output = value; }};
  }

  template <typename T>
  static inline OptionHandler createWithList(std::vector<T> &output) {
    return OptionHandler{[&output](poplar::StringRef values) {
      std::istringstream iss(values);
      for (std::string element; std::getline(iss, element, ',');) {
        T value;
        std::istringstream elementStream(element);
        if (element.find("0x") != std::string::npos) {
          elementStream >> std::hex >> value;
        } else {
          elementStream >> std::dec >> value;
        }
        if (elementStream.fail() || !elementStream.eof()) {
          throw poplar::invalid_option("Not a comma-separated list of "
                                       "integers");
        }
        output.emplace_back(value);
      }
    }};
  }
};

class OptionSpec {
  using value_type = std::pair<const std::string, OptionHandler>;
  using map_type = std::map<const std::string, OptionHandler>;
  using initializer_list_t = std::initializer_list<value_type>;
  map_type handlers;

public:
  OptionSpec(initializer_list_t &&handlers) : handlers(std::move(handlers)) {}

  // Validate and handle options based on the spec
  void parse(poplar::StringRef option, poplar::StringRef value,
             bool ignoreUnknown = false) const {
    auto it = handlers.find(option);
    if (it == handlers.end()) {
      if (ignoreUnknown) {
        return;
      } else {
        std::stringstream s;
        s << "Unrecognised option '" << option << "'";
        throw poplar::invalid_option(s.str());
      }
    }
    try {
      const auto &handler = it->second;
      handler.parseValue(value);
    } catch (const poplar::invalid_option &e) {
      std::stringstream s;
      s << "Invalid value '" << value << "'"
        << " for option '" << option << "': " << e.what();
      throw poplar::invalid_option(s.str());
    }
  }
};

} // end namespace poplibs

#endif // poplibs_support_OptionParsing_hpp
