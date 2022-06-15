// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef poplibs_test_ProgressBar_hpp
#define poplibs_test_ProgressBar_hpp

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include <cassert>
#include <chrono>
#include <iostream>

// A simple textual progress bar.
struct ProgressBar {
  ProgressBar(const char *title, size_t max, bool is_enabled = true,
              std::ostream &os = std::cout)
      : title(title), max_value(max), enabled(is_enabled), out(os) {
    buffer.resize(max_width);
    std::fill(buffer.begin(), buffer.end(), blankChar);
  }
  void update(size_t count) {
    assert(count <= max_value);
    assert(count >= last_count);
    if (!enabled)
      return;
    if (last_count == 0)
      start = clock::now();
    double ratio = (double)count / max_value;
    size_t old_size = max_width * (double)last_count / max_value;
    size_t size = max_width * ratio;
    std::fill(buffer.begin() + old_size, buffer.begin() + size, fillChar);
    if (size < max_width)
      buffer[size] = '>';
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now() - start);
    fmt::print(out, "\r{}[{}] ({:.3}%) (took {:.3} s)  ", title, buffer,
               ratio * 100, duration.count() / 1000.0);
    if (count == max_value) // Finished.
      fmt::print(out, "\n");
    last_count = count;
  }
  ProgressBar &operator++() {
    update(last_count + 1);
    return *this;
  }

private:
  const char *title = nullptr;
  size_t max_value = 100;
  bool enabled = true;
  std::ostream &out;
  size_t last_count = 0;
  std::string buffer;
  constexpr static char fillChar = '=';
  constexpr static char blankChar = ' ';
  constexpr static size_t max_width = 44;
  using clock = std::chrono::steady_clock;
  clock::time_point start;
};

#endif
