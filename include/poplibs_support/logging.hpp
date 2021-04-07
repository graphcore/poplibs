// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_logging_logging_hpp
#define poplibs_support_logging_logging_hpp

// print.hpp must be included before fmt.hpp so that containers can
// be logged successfully on macos
#include "poplibs_support/print.hpp"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include <string>

/// This is a simple logging system for Poplibs based on spdlog. The easiest
/// way to use it is to simply call `logging::<module>::<level>()` where <level>
/// is one of trace, debug, info, warn or err. For example:
///
///   #include <core/logging/logging.hpp>
///
///   void foo(int i) {
///     logging::poplin::info("foo({}) called", i);
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

enum class Module {
  popfloat = 0,
  poplin = 1,
  popnn = 2,
  popops = 3,
  poprand = 4,
  popsolver = 5,
  popsparse = 6,
  poputil = 7,
  poplibs = 8, // Default - deprecated

  size = 9
};

// Set the current log level to one of the above levels. The default
// log level is set by the POPLIBS_LOG_LEVEL environment variable
// and is off by default.
void setLogLevel(Module m, Level l);

// Return true if the passed log level is currently enabled.
bool shouldLog(Module m, Level l);

// Flush the log. By default it is only flushed when the underlying libc
// decides to.
void flush(Module m);

// Log a message. You should probably use the MAKE_LOG_TEMPLATE macros
// instead, e.g. logging::poplin::debug("A debug message").
void log(Module m, Level l, std::string &&msg);

// Log a formatted message. This uses the `fmt` C++ library for formatting.
// See https://github.com/fmtlib/fmt for details. You should probably use
// the MAKE_LOG_TEMPLATE macros instead, e.g.
// logging::poplin::debug("The answer is: {}", 42).
template <typename... Args>
void log(Module m, Level l, const char *s, Args &&...args) {
  // Avoid formatting if the logging is disabled anyway.
  if (shouldLog(m, l)) {
    log(m, l, fmt::format(s, std::forward<Args>(args)...));
  }
}

// Create functions of the form logging::session::debug("Msg").
// where session if the name of the log module and debug is the
// logging level
#define MAKE_MODULE_LOG_TEMPLATE(fnName, module, lvl)                          \
  template <typename... Args>                                                  \
  inline void fnName(const std::string &s, Args &&...args) {                   \
    log(Module::module, Level::lvl, s.c_str(), std::forward<Args>(args)...);   \
  }                                                                            \
  template <typename... Args>                                                  \
  inline void fnName(const char *s, Args &&...args) {                          \
    log(Module::module, Level::lvl, s, std::forward<Args>(args)...);           \
  }

#define MAKE_MODULE_TEMPLATE(MODULE)                                           \
  namespace MODULE {                                                           \
  MAKE_MODULE_LOG_TEMPLATE(trace, MODULE, Trace)                               \
  MAKE_MODULE_LOG_TEMPLATE(debug, MODULE, Debug)                               \
  MAKE_MODULE_LOG_TEMPLATE(info, MODULE, Info)                                 \
  MAKE_MODULE_LOG_TEMPLATE(warn, MODULE, Warn)                                 \
  MAKE_MODULE_LOG_TEMPLATE(err, MODULE, Err)                                   \
  inline void flush() { flush(Module::MODULE); }                               \
  inline void setLogLevel(Level l) { setLogLevel(Module::MODULE, l); }         \
  inline bool shouldLog(Level l) { return shouldLog(Module::MODULE, l); }      \
  template <typename... Args>                                                  \
  void log(Level l, const char *s, Args &&...args) {                           \
    log(Module::MODULE, l, fmt::format(s, std::forward<Args>(args)...));       \
  }                                                                            \
  }

// The definition of the logging modules
MAKE_MODULE_TEMPLATE(popfloat)
MAKE_MODULE_TEMPLATE(poplin)
MAKE_MODULE_TEMPLATE(popnn)
MAKE_MODULE_TEMPLATE(popops)
MAKE_MODULE_TEMPLATE(poprand)
MAKE_MODULE_TEMPLATE(popsolver)
MAKE_MODULE_TEMPLATE(popsparse)
MAKE_MODULE_TEMPLATE(poputil)
MAKE_MODULE_TEMPLATE(poplibs)

#undef MAKE_MODULE_LOG_TEMPLATE
#undef MAKE_MODULE_TEMPLATE

} // namespace logging
} // namespace poplibs_support

#endif // poplibs_support_logging_logging_hpp
