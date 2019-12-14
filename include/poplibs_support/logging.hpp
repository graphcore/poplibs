// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef poplibs_support_logging_logging_hpp
#define poplibs_support_logging_logging_hpp

#include <string>
// print.hpp must be included before fmt.hpp so that containers can
// be logged successfully on macos
#include "poplibs_support/print.hpp"
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

/// This is a simple logging system for Poplibs based on spdlog. The easiest
/// way to use it is to simply call `logging::<level>()` where <level> is one
/// of trace, debug, info, warn or err. For example:
///
///   #include <core/logging/logging.hpp>
///
///   void foo(int i) {
///     logging::info("foo({}) called", i);
///   }
///
/// logging can be configured by the methods below, or by environment
/// variables, eg
/// POPLIBS_LOG_LEVEL=ERR
/// POPLIBS_LOG_DEST=Mylog.txt
///
/// Formatting is done using the `fmt` library. It supports {}-style and %-style
/// format specification strings. See https://github.com/fmtlib/fmt for details.

namespace poplibs_support {
namespace logging {

// level 5 is "critical" in spglog, which we don't use so isn't exposed here.
enum class Level {
  /// Trace level is used for per-tile and other very noisy debug information.
  Trace = 0,
  /// Debug level is used for extra information that is typically per-graph.
  Debug = 1,
  /// Info level is used to provide a high level overview of which public
  /// function is called with what shapes of tensors.
  Info = 2,
  /// Warn level is used when an invariant wasn't able to be met (like running
  /// out of memory) but the compilation process can continue.
  Warn = 3,
  /// Err level is used for hard errors, they normally immediately precede the
  /// compilation finishing with an error.
  Err = 4,
  Off = 6,
};

// Set the current log level to one of the above levels. The default
// log level is set by the POPLIBS_LOG_LEVEL environment variable
// and is off by default.
void setLogLevel(Level l);

// Return true if the passed log level is currently enabled.
bool shouldLog(Level l);

// Flush the log. By default it is only flushed when the underlying libc
// decides to.
void flush();

// Log a message. You should probably use the MAKE_LOG_TEMPLATE macros
// instead, e.g. logging::debug("A debug message").
void log(Level l, std::string &&msg);

// Log a formatted message. This uses the `fmt` C++ library for formatting.
// See https://github.com/fmtlib/fmt for details. You should probably use
// the MAKE_LOG_TEMPLATE macros instead, e.g.
// logging::debug("The answer is: {}", 42).
template <typename... Args>
void log(Level l, const char *s, const Args &... args) {
  // Avoid formatting if the logging is disabled anyway.
  if (shouldLog(l)) {
    log(l, fmt::format(s, args...));
  }
}

// Create a bit of syntactic sugar which allows log statements
// of the form logging::debug("Msg").
#define MAKE_LOG_TEMPLATE(fnName, lvl)                                         \
  template <typename... Args>                                                  \
  inline void fnName(const char *s, const Args &... args) {                    \
    log(Level::lvl, s, std::forward<const Args>(args)...);                     \
  }                                                                            \
  template <typename... Args>                                                  \
  inline void fnName(const std::string &s, const Args &... args) {             \
    log(Level::lvl, s.c_str(), std::forward<const Args>(args)...);             \
  }

MAKE_LOG_TEMPLATE(trace, Trace)
MAKE_LOG_TEMPLATE(debug, Debug)
MAKE_LOG_TEMPLATE(info, Info)
MAKE_LOG_TEMPLATE(warn, Warn)
MAKE_LOG_TEMPLATE(err, Err)

// Convenience macro to create a log entry prefixed with function name e.g.:
//    void someFunc(int i) {
//      FUNC_LOGGER(info, " with i := {}", i);
//    }
// Then the log entry would be something like:
// 14:30:31.00 [I] void someFunc(int): with i := 42
// NOTE: Because of the limitations of __VA_ARGS__ this log entry must have at
// least one parameter.
#define FUNC_LOGGER(lvl, fmtStr, ...)                                          \
  logging::lvl("{}: " fmtStr, __PRETTY_FUNCTION__, __VA_ARGS__)

} // namespace logging
} // namespace poplibs_support

#endif // poplibs_support_logging_logging_hpp
