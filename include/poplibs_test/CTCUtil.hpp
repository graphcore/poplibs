// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_ctc_util_hpp
#define poplibs_test_ctc_util_hpp

#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>

#include <random>
#include <vector>

namespace poplibs_test {
namespace ctc {

class RandomUtil {
  std::mt19937 gen;
  template <typename T> static void checkMinMax(T min, T max) {
    if (max < min) {
      poputil::poplibs_error(
          "max must be greater than min when specifying random range");
    }
  }

public:
  RandomUtil(unsigned s) { gen.seed(s); }

  template <typename T> T range(T min, T max) {
    std::uniform_int_distribution<> range(min, max);
    return range(gen);
  }

  template <typename T> std::function<T()> generator(T min, T max) {
    checkMinMax(min, max);
    if constexpr (std::is_integral<T>::value) {
      return [&, min, max]() -> T {
        std::uniform_int_distribution<T> range(min, max);
        return range(gen);
      };
    } else {
      return [&, min, max]() -> T {
        std::uniform_real_distribution<T> range(min, max);
        return range(gen);
      };
    }
  }
};

// Converts input sequence to ctc loss extendedLabels version, e.g.
//   `a` -> `- a -`
//   `aabb` -> `- a - b -`
// Optionally remove duplicates then insert blanks
std::vector<unsigned> extendedLabels(const std::vector<unsigned> &input,
                                     unsigned blankSymbol,
                                     bool stripDuplicates = false);

void print(const std::vector<unsigned> &sequence);
void print(const std::vector<unsigned> &sequence, unsigned blank);

template <typename FPType>
std::pair<boost::multi_array<FPType, 2>, std::vector<unsigned>>
provideInputWithPath(unsigned labelLength, unsigned timeSteps, unsigned maxT,
                     unsigned numClasses, unsigned blankClass,
                     RandomUtil &rand);

} // namespace ctc
} // namespace poplibs_test

#endif // poplibs_test_ctc_util_hpp
