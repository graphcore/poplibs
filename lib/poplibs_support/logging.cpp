// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/logging.hpp"
#include <array>
#include <iostream>
#include <poplar/exceptions.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace poplibs_support {
namespace logging {

namespace {

// Check our enums match (incase spdlog changes under us)
static_assert(static_cast<spdlog::level::level_enum>(Level::Trace) ==
                  spdlog::level::trace,
              "Logging enum mismatch");
static_assert(static_cast<spdlog::level::level_enum>(Level::Off) ==
                  spdlog::level::off,
              "Logging enum mismatch");

// Translate to a speedlog log level.
spdlog::level::level_enum translate(Level l) {
  return static_cast<spdlog::level::level_enum>(l);
}

// Stores the logging object needed by spdlog.
struct LoggingContext {
public:
  static spdlog::logger &getLogger(Module m);

private:
  LoggingContext();
  static LoggingContext &instance() {
    // This avoids the static initialisation order fiasco, but doesn't solve the
    // deinitialisation order. Who logs in destructors anyway?
    static LoggingContext loggingContext;
    return loggingContext;
  }

  std::shared_ptr<spdlog::sinks::sink> sink;
  std::array<std::unique_ptr<spdlog::logger>,
             static_cast<std::size_t>(Module::size)>
      loggers;
};

Level logLevelFromString(const std::string &level) {

  if (level == "TRACE")
    return Level::Trace;
  if (level == "DEBUG")
    return Level::Debug;
  if (level == "INFO")
    return Level::Info;
  if (level == "WARN")
    return Level::Warn;
  if (level == "ERR")
    return Level::Err;
  if (level == "OFF" || level == "")
    return Level::Off;

  throw poplar::runtime_error(
      "Unknown POPLIBS_LOG_LEVEL '" + level +
      "'. Valid values are TRACE, DEBUG, INFO, WARN, ERR and OFF.");
}

std::string moduleName(Module m) {
  switch (m) {
  case Module::popfloat:
    return "POPFLOAT";
  case Module::poplin:
    return "POPLIN";
  case Module::popnn:
    return "POPNN";
  case Module::popops:
    return "POPOPS";
  case Module::poprand:
    return "POPRAND";
  case Module::popsolver:
    return "POPSOLVER";
  case Module::popsparse:
    return "POPSPARSE";
  case Module::poputil:
    return "POPUTIL";
  default:
    return "POPLIBS";
  }
}

static std::size_t getModuleMaxLength(Module m) {
  constexpr std::array modules = {
      Module::popfloat, Module::poplin,    Module::popnn,     Module::popops,
      Module::poprand,  Module::popsolver, Module::popsparse, Module::poputil};
  std::size_t maxLength = 0;
  for (const auto module : modules) {
    maxLength = std::max(maxLength, moduleName(module).length());
  }
  return maxLength;
}

template <typename Mutex>
void setColours(spdlog::sinks::ansicolor_sink<Mutex> &sink) {
  // See https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
  static const std::string brightBlack = "\033[90m";

  // info is unset so that it uses the system default colour.
  sink.set_color(spdlog::level::trace, brightBlack);
  sink.set_color(spdlog::level::debug, sink.cyan);
  sink.set_color(spdlog::level::warn, sink.yellow);
  sink.set_color(spdlog::level::err, sink.red_bold);
}

LoggingContext::LoggingContext() {
  auto POPLIBS_LOG_DEST = std::getenv("POPLIBS_LOG_DEST");
  auto POPLIBS_LOG_LEVEL = std::getenv("POPLIBS_LOG_LEVEL");

  // Get logging output from the POPLIBS_LOG_DEST environment variable.
  // The valid options are "stdout", "stderr", or if it is neither
  // of those it is treated as a filename. The default is stderr.
  std::string logDest = POPLIBS_LOG_DEST ? POPLIBS_LOG_DEST : "stderr";

  if (logDest == "stdout") {
    auto colouredSink =
        std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
    setColours(*colouredSink);
    sink = colouredSink;
  } else if (logDest == "stderr") {
    auto colouredSink =
        std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
    setColours(*colouredSink);
    sink = colouredSink;
  } else {
    try {
      sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logDest, true);
    } catch (const spdlog::spdlog_ex &e) {
      std::cerr << "Error opening log file: " << e.what() << std::endl;
      throw;
    }
  }

  // Get logging level from OS ENV. The default level is off.
  const auto defaultLevel =
      logLevelFromString(POPLIBS_LOG_LEVEL ? POPLIBS_LOG_LEVEL : "OFF");

  const auto createLogger = [&](Module m) {
    const auto getLogLevelForModule = [&](Module m) {
      const auto envVar = "POPLIBS_" + moduleName(m) + "_LOG_LEVEL";
      const auto value = std::getenv(envVar.c_str());
      if (value) {
        return logLevelFromString(value);
      } else {
        return defaultLevel;
      }
    };

    auto logger = std::make_unique<spdlog::logger>(moduleName(m), sink);
    logger->set_level(translate(getLogLevelForModule(m)));
    const auto n = std::to_string(getModuleMaxLength(m));
    logger->set_pattern("%^%Y-%m-%dT%H:%M:%S.%fZ PL:%-" + n +
                            "n %P.%t %L: %v%$",
                        spdlog::pattern_time_type::utc);
    loggers[static_cast<std::size_t>(m)] = std::move(logger);
  };

  createLogger(Module::popfloat);
  createLogger(Module::poplin);
  createLogger(Module::popnn);
  createLogger(Module::popops);
  createLogger(Module::poprand);
  createLogger(Module::popsolver);
  createLogger(Module::popsparse);
  createLogger(Module::poputil);
}

spdlog::logger &LoggingContext::getLogger(Module m) {
  return *LoggingContext::instance().loggers.at(static_cast<std::size_t>(m));
}

} // namespace

void log(Module m, Level l, std::string &&msg) {
  LoggingContext::getLogger(m).log(translate(l), std::move(msg));
}

bool shouldLog(Module m, Level l) {
  return LoggingContext::getLogger(m).should_log(translate(l));
}

void setLogLevel(Module m, Level l) {
  LoggingContext::getLogger(m).set_level(translate(l));
}

void flush(Module m) { LoggingContext::getLogger(m).flush(); }

} // namespace logging
} // namespace poplibs_support
