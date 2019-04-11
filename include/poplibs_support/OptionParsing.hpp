#ifndef poplibs_support_OptionParsing_hpp
#define poplibs_support_OptionParsing_hpp

#include <cassert>
#include <poplar/exceptions.hpp>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <string>
#include <sstream>

namespace poplibs {

// OptionSpec/OptionHandler used to build up a specification of what
// options and their values should be, and to translate the value strings
// to real values.
class OptionHandler {
  std::string valueDesc;
  std::function<bool(const std::string&)> valueHandler;
public:
  template <typename T1, typename T2>
  OptionHandler(T1 &&valueDesc, T2 &&valueHandler) :
    valueDesc(std::forward<T1>(valueDesc)),
    valueHandler(std::forward<T2>(valueHandler)) {}
  const std::string &getValueDesc() const { return valueDesc; }
  bool parseValue(const std::string &value) const {
    return valueHandler(value);
  }

  // Utility functions to help build a spec
  template <typename T>
  static inline std::string
  describeEnumValues(const std::map<std::string, T> &valueMap) {
    std::stringstream s;
    assert(valueMap.size() > 0);
    auto it = valueMap.begin();
    s << "One of '" << it->first << "'";
    while(++it != valueMap.end()) {
      s << ", '"<< it->first << "'";
    }
    return s.str();
  }

  template <typename T, typename ValueMapT = std::map<std::string, T>>
  static inline OptionHandler
  createWithEnum(T &output, ValueMapT &&valueMap) {
    auto handler = [&output](const ValueMapT &map, const std::string &value) {
      auto it = map.find(value);
      if (it == map.end())
        return false;
      output = it->second;
      return true;
    };

    // Allow forwarding of rvalues for valueMap but we always want
    // to hold a copy for safety. C++14's generalised lambda captures
    // would simplify this: [map=std::forward<ValueMapT>(valueMap),...
    using namespace std::placeholders;
    return OptionHandler{
      describeEnumValues(valueMap),
      std::bind(std::move(handler), std::forward<ValueMapT>(valueMap), _1)
    };
  }

  static inline OptionHandler
  createWithBool(bool &output) {
    static std::map<std::string, bool> boolMap{
      { "true", true },
      { "false", false }
    };
    return createWithEnum(output, boolMap);
  }

  static inline OptionHandler
  createWithUnsignedInt(unsigned &output,
                        unsigned lowerBound = 0,
                        unsigned upperBound
                          = std::numeric_limits<unsigned>::max()) {
    std::stringstream valueDesc;
    valueDesc << "Unsigned 32-bit integers";
    if (lowerBound != 0 ||
        upperBound != std::numeric_limits<unsigned>::max()) {
      valueDesc << " in the range [" << lowerBound << "," << ')';
    }
    return OptionHandler{
      valueDesc.str(),
      [lowerBound, upperBound, &output](const std::string &value) {
        std::stringstream s(value);
        unsigned val;
        s >> val;
        if (s.fail())
          return false;
        if (val < lowerBound || val >= upperBound)
          return false;
        output = val;
        return true;
      }
    };
  }

  static inline OptionHandler
  createWithDouble(double &output) {
    return OptionHandler{
      "Floating point values",
      [&output](const std::string &value) {
        std::stringstream s(value);
        double val;
        s >> val;
        if (s.fail())
          return false;
        output = val;
        return true;
      }
    };
  }
};

class OptionSpec {
  using value_type = std::pair<const std::string, OptionHandler>;
  using map_type = std::map<const std::string, OptionHandler>;
  using initializer_list_t = std::initializer_list<value_type>;
  map_type handlers;
public:
  OptionSpec(initializer_list_t &&handlers) :
    handlers(std::move(handlers)) {}

  // Validate and handle options based on the spec
  void parse(const std::string &option, const std::string &value) const {
    using poplar::invalid_option;
    auto it = handlers.find(option);
    if (it == handlers.end()) {
      std::stringstream s;
      s << "Unrecognised option '" << option << "'";
      throw invalid_option(s.str());
    }
    const auto &handler = it->second;
    if (!handler.parseValue(value)) {
      std::stringstream s;
      s << "Invalid value '" << value << "'"
        << " for option '" << option << "'. "
        << "Valid values: " << handler.getValueDesc();
      throw invalid_option(s.str());
    }
  }
};

} // end namespace poplibs

#endif // poplibs_support_OptionParsing_hpp
