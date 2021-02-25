// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include <boost/random.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <cmath>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;

namespace poplibs_test {
namespace util {

std::unique_ptr<char[]>
allocateHostMemoryForTensor(const Target &target, const Tensor &t,
                            unsigned replicationFactor,
                            std::size_t &allocatedSizeInBytes) {
  const auto dType = t.elementType();
  std::unique_ptr<char[]> p;

  allocatedSizeInBytes =
      t.numElements() * target.getTypeSize(dType) * replicationFactor;

  p.reset(new char[allocatedSizeInBytes]);
  std::fill(&p[0], &p[allocatedSizeInBytes], 0);

  return p;
}

std::unique_ptr<char[]>
allocateHostMemoryForTensor(const Target &target, const Tensor &t,
                            unsigned replicationFactor) {
  std::size_t allocatedSizeInbytes = 0;

  return allocateHostMemoryForTensor(target, t, replicationFactor,
                                     allocatedSizeInbytes);
}

std::unique_ptr<char[]>
allocateHostMemoryForTensor(const Tensor &t, const std::string &name,
                            Graph &graph,
                            boost::optional<Sequence &> uploadProg,
                            boost::optional<Sequence &> downloadProg,
                            std::vector<std::pair<std::string, char *>> &map) {
  std::unique_ptr<char[]> p = allocateHostMemoryForTensor(
      graph.getTarget(), t, graph.getReplicationFactor());

  if (downloadProg) {
    auto downloadId = graph.addDeviceToHostFIFO(
        name + "_download", t.elementType(), t.numElements());
    downloadProg->add(Copy(t, downloadId, true, name));
    map.emplace_back(name + "_download", p.get());
  }

  if (uploadProg) {
    auto uploadId = graph.addHostToDeviceFIFO(name + "_upload", t.elementType(),
                                              t.numElements());
    uploadProg->add(Copy(uploadId, t, true, name));
    map.emplace_back(name + "_upload", p.get());
  }

  return p;
}

void attachStreams(Engine &e,
                   const std::vector<std::pair<std::string, char *>> &map) {
  for (const auto &p : map) {
    e.connectStream(p.first, p.second);
  }
}

template <typename T>
void roundToHalfPrecision(const Target &target, T *begin, T *end) {
  auto N = end - begin;
  std::vector<char> buf(N * target.getTypeSize(poplar::HALF));
  detail::copyToDevice(target, begin, buf.data(), N);
  detail::copyFromDevice(target, buf.data(), begin, N);
}

template <typename T, typename F>
void writeValues(T *begin, T *end, F generator) {
  for (auto it = begin; it != end; ++it) {
    *it = generator();
  }
}

template <typename T>
void writeRandomBinaryValues(const Target &target, const Type &type, T *begin,
                             T *end, T a, T b, std::mt19937 &randomEngine) {
  if (type == poplar::FLOAT || type == poplar::HALF) {
    boost::random::bernoulli_distribution<> dist{};
    writeValues(begin, end, [&]() { return dist(randomEngine) ? a : b; });
    if (type == poplar::HALF) {
      roundToHalfPrecision(target, begin, end);
    }
  } else {
    throw poputil::poplibs_error("Unsupported type");
  }
}

template void writeRandomBinaryValues<double>(const Target &target,
                                              const Type &type, double *begin,
                                              double *end, double a, double b,
                                              std::mt19937 &randomEngine);
template void writeRandomBinaryValues<unsigned>(const Target &target,
                                                const Type &type,
                                                unsigned *begin, unsigned *end,
                                                unsigned a, unsigned b,
                                                std::mt19937 &randomEngine);

template <typename T>
void writeRandomValues(const Target &target, const Type &type, T *begin, T *end,
                       T min, T max, std::mt19937 &randomEngine) {
  if (type == poplar::FLOAT || type == poplar::HALF) {
    boost::random::uniform_real_distribution<> dist(min, max);
    writeValues(begin, end, [&]() { return dist(randomEngine); });
    if (type == poplar::HALF) {
      roundToHalfPrecision(target, begin, end);
    }
  } else if (type == poplar::INT) {
    boost::random::uniform_int_distribution<int> dist(min, max);
    writeValues(begin, end, [&]() { return dist(randomEngine); });
  } else if (type == poplar::UNSIGNED_INT) {
    boost::random::uniform_int_distribution<unsigned> dist(min, max);
    writeValues(begin, end, [&]() { return dist(randomEngine); });
  } else if (type == poplar::BOOL) {
    boost::random::uniform_int_distribution<unsigned> dist(0, 1);
    writeValues(begin, end, [&]() { return dist(randomEngine); });
  } else {
    throw poputil::poplibs_error("Unknown type");
  }
}

template void writeRandomValues<double>(const Target &target, const Type &type,
                                        double *begin, double *end, double min,
                                        double max, std::mt19937 &randomEngine);
template void writeRandomValues<unsigned>(const Target &target,
                                          const Type &type, unsigned *begin,
                                          unsigned *end, unsigned min,
                                          unsigned max,
                                          std::mt19937 &randomEngine);

template void writeRandomValues<int>(const Target &target, const Type &type,
                                     int *begin, int *end, int min, int max,
                                     std::mt19937 &randomEngine);

size_t maxContiguousInteger(const Type &t) {
  if (t == HALF)
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    return 2048;
  else if (t == FLOAT)
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
    return 16777216;
  else if (t == INT)
    return std::numeric_limits<int>::max();
  else
    throw std::runtime_error("Type not supported");
}

size_t maxContiguousIntegerFromBinaryOp(const Type &inputType,
                                        const Type &outputType) {
  return std::min(maxContiguousInteger(inputType),
                  maxContiguousInteger(outputType));
}

bool isLikelyToHaveNumericalErrorsUsingBernoulli(size_t macsPerOutputElement,
                                                 const Type &inputType,
                                                 const Type &outputType) {
  // Naively increasing maxInt by a factor of 30 is still very unlikely to
  // over/underflow. Since it's so unlikely, we'll pad this value a little.
  const size_t maxInt = maxContiguousIntegerFromBinaryOp(inputType, outputType);
  return macsPerOutputElement > 10 * maxInt;
}

template <typename FPType>
bool checkIsClose(FPType a, FPType b, double relativeTolerance) {
  // These checks are necessary because close_at_tolerance doesn't handle
  // NaN, infinity or comparing against zero.
  if (a == b) {
    return true;
  }
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  boost::math::fpc::small_with_tolerance<FPType> nearZero(relativeTolerance);
  if (nearZero(a) && nearZero(b)) {
    return true;
  }
  if (!boost::math::fpc::close_at_tolerance<FPType>(relativeTolerance)(a, b)) {
    return false;
  }

  return true;
}

template bool checkIsClose<bool>(bool, bool, double);
template bool checkIsClose<int>(int, int, double);
template bool checkIsClose<unsigned>(unsigned, unsigned, double);
template bool checkIsClose<float>(float, float, double);
template bool checkIsClose<double>(double, double, double);

std::string prettyCoord(const std::string &name, std::size_t index,
                        const std::vector<std::size_t> &shape) {
  std::string str = name + "[";
  auto N = std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                           std::multiplies<size_t>());
  for (unsigned i = 0; i != shape.size(); ++i) {
    N = N / shape[i];
    if (i != 0)
      str = str += ",";
    str = str += std::to_string(index / N);
    index = index % N;
  }
  str += "]";
  return str;
}

template <typename intType>
bool checkEqual(const std::string &name, const intType *actual,
                const std::vector<std::size_t> &shape, const intType *expected,
                std::size_t N) {
  auto it = actual;
  auto end = it + N;
  bool equal = true;
  unsigned mismatch = 0;
  constexpr unsigned mismatchPrinted = 20;
  for (; it != end; ++it, ++expected) {
    if (*it != *expected) {
      equal = false;
      if (mismatch++ < mismatchPrinted) {
        std::cerr << "mismatch on element ";
        const auto n = it - actual;
        std::cerr << prettyCoord(name, n, shape) << ':';
        std::cerr << " expected=" << *expected;
        std::cerr << " actual=" << *it << '\n';
      }
    }
  }
  if (mismatch >= mismatchPrinted) {
    std::cerr << "Number of mismatches " << mismatch << " exceeded "
              << mismatchPrinted << ", messages suppressed.\n";
  }
  return equal;
}

template bool checkEqual<unsigned>(const std::string &, const unsigned *,
                                   const std::vector<std::size_t> &,
                                   const unsigned *, std::size_t);

template bool checkEqual<std::uint64_t>(const std::string &,
                                        const std::uint64_t *,
                                        const std::vector<std::size_t> &,
                                        const std::uint64_t *, std::size_t);

template <typename FPType>
bool checkIsClose(const std::string &name, const FPType *actual,
                  const std::vector<std::size_t> &shape, const FPType *expected,
                  std::size_t N, double relativeTolerance,
                  double absoluteTolerance) {
  auto it = actual;
  auto end = it + N;
  bool isClose = true;
  unsigned mismatch = 0;
  unsigned mismatchPrinted = 20;
  for (; it != end; ++it, ++expected) {
    if ((std::isnan(*it) || std::isnan(*expected)) ||
        (!checkIsClose(*it, *expected, relativeTolerance) &&
         std::fabs(*expected - *it) > absoluteTolerance)) {
      isClose = false;
      if (mismatch++ < mismatchPrinted) {
        const auto n = it - actual;
        std::cerr << "mismatch on element ";
        std::cerr << prettyCoord(name, n, shape) << ':';
        std::cerr << " expected=" << *expected;
        std::cerr << " actual=" << *it;
        std::cerr << " (abs=" << *it - *expected;
        std::cerr << ", rel=" << ((*it / *expected) - 1.0) * 100.0 << "%)\n";
      }
    }
  }
  if (mismatch >= mismatchPrinted) {
    std::cerr << "Number of mismatches " << mismatch << " exceeded "
              << mismatchPrinted << ", messages suppressed.\n";
  }
  return isClose;
}

template bool checkIsClose<float>(const std::string &, const float *,
                                  const std::vector<std::size_t> &,
                                  const float *, std::size_t, double, double);

template bool checkIsClose<double>(const std::string &, const double *,
                                   const std::vector<std::size_t> &,
                                   const double *, std::size_t, double, double);

void addGlobalExchangeConstraints(IPUModel &ipuModel) {
  const auto numIPUs = ipuModel.numIPUs;
  if (numIPUs == 1)
    return;
  if (numIPUs % 2 != 0) {
    throw runtime_error("IPU modeling does not support an odd number "
                        "of IPUs");
  }
  // Calculate the set of (src IPU, dst IPU) pairs whose traffic
  // is routed through each link by tracing the route every possible
  // (src IPU, dst IPU) pair takes through the network.
  std::vector<std::vector<GlobalExchangeFlow>> intraIpuFlows(numIPUs);
  std::vector<std::vector<GlobalExchangeFlow>> crossRoutingFlows(numIPUs);
  std::vector<std::vector<GlobalExchangeFlow>> upFlows(numIPUs);
  std::vector<std::vector<GlobalExchangeFlow>> downFlows(numIPUs);
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    for (unsigned otherIpu = 0; otherIpu != numIPUs; ++otherIpu) {
      if (otherIpu == ipu)
        continue;
      if (ipu / 2 == otherIpu / 2) {
        intraIpuFlows[ipu].emplace_back(ipu, otherIpu);
      } else {
        unsigned currentIpu = ipu;
        // If the destination is on the other side of the ladder we first
        // route across the rung of the ladder.
        if (currentIpu % 2 != otherIpu % 2) {
          crossRoutingFlows[currentIpu].emplace_back(ipu, otherIpu);
          currentIpu = currentIpu % 2 == 0 ? currentIpu + 1 : currentIpu - 1;
        }
        // We now route up or down depending on the destination IPU number.
        bool routeUp = currentIpu < otherIpu;
        do {
          if (routeUp) {
            upFlows[currentIpu].emplace_back(ipu, otherIpu);
            currentIpu += 2;
          } else {
            downFlows[currentIpu].emplace_back(ipu, otherIpu);
            currentIpu -= 2;
          }
        } while (currentIpu != otherIpu);
      }
    }
  }
  // Link speed in bits per second.
  double linkBandwidth = 128.0 * 1024 * 1024 * 1024;
  double linkEfficiency = 0.85;
  const unsigned numIntraIpuLinks = 4;
  const unsigned numCrossRoutingLinks = 2;
  const unsigned numUpLinks = 2;
  const unsigned numDownLinks = 2;
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    if (!intraIpuFlows[ipu].empty()) {
      ipuModel.globalExchangeConstraints.push_back(GlobalExchangeConstraint(
          numIntraIpuLinks * linkBandwidth * linkEfficiency,
          intraIpuFlows[ipu]));
    }
    if (!crossRoutingFlows[ipu].empty()) {
      ipuModel.globalExchangeConstraints.push_back(GlobalExchangeConstraint(
          numCrossRoutingLinks * linkBandwidth * linkEfficiency,
          crossRoutingFlows[ipu]));
    }
    if (!upFlows[ipu].empty()) {
      ipuModel.globalExchangeConstraints.push_back(GlobalExchangeConstraint(
          numUpLinks * linkBandwidth * linkEfficiency, upFlows[ipu]));
    }
    if (!downFlows[ipu].empty()) {
      ipuModel.globalExchangeConstraints.push_back(GlobalExchangeConstraint(
          numDownLinks * linkBandwidth * linkEfficiency, downFlows[ipu]));
    }
  }
}

void setGlobalSyncLatency(IPUModel &ipuModel) {
  // One hop within a card plus one hop per card pair.
  unsigned numHops = 1 + ((ipuModel.numIPUs / 2) / 2);
  const double syncLatencyPerHop = 15e-9;
  ipuModel.globalSyncCycles =
      std::ceil(syncLatencyPerHop * ipuModel.tileClockFrequency * numHops * 2);
}

Tensor createGenericConvInput(Graph &graph, const Type &type,
                              std::size_t batchSize, std::size_t numConvGroups,
                              std::size_t chansPerConvGroup,
                              const std::vector<std::size_t> &fieldShape,
                              const std::string &name) {
  assert(type == HALF || type == FLOAT);
  bool isFloat = type == FLOAT;
  std::size_t convGroupsPerGroup, chansPerGroup;
  // Take an educated guess at how the input is grouped. We assume
  // the input is laid out for AMP unless the number of input channels
  // is small in which case we assume it is laid out correctly for SLIC
  if (type == HALF && (chansPerConvGroup == 1 || chansPerConvGroup == 2) &&
      chansPerConvGroup * numConvGroups % 4 == 0) {
    convGroupsPerGroup = 4 / chansPerConvGroup;
    chansPerGroup = chansPerConvGroup;
  } else {
    convGroupsPerGroup = 1;
    auto weightsPerConvUnit = graph.getTarget().getWeightsPerConvUnit(isFloat);
    chansPerGroup =
        gcd(static_cast<std::size_t>(weightsPerConvUnit), chansPerConvGroup);
  }
  std::vector<std::size_t> tensorShape = {numConvGroups / convGroupsPerGroup,
                                          chansPerConvGroup / chansPerGroup,
                                          batchSize};
  tensorShape.insert(tensorShape.end(), fieldShape.begin(), fieldShape.end());
  tensorShape.push_back(convGroupsPerGroup);
  tensorShape.push_back(chansPerGroup);

  auto t = graph.addVariable(type, tensorShape, name);
  const auto vectorWidth = graph.getTarget().getVectorWidth(type);
  const auto grainSize = lcm(
      static_cast<unsigned>(chansPerGroup * convGroupsPerGroup), vectorWidth);
  poputil::mapTensorLinearly(graph, t, 0, grainSize);
  return t.dimShufflePartial({2, t.rank() - 2, t.rank() - 1}, {0, 2, 4})
      .reshapePartial(1, 5, {numConvGroups * chansPerConvGroup});
}

Tensor createGenericFullyConnectedInput(Graph &graph, const Type &type,
                                        std::size_t numGroups,
                                        std::size_t batchSize,
                                        std::size_t inputSize,
                                        const std::string &name) {
  auto convInput = createGenericConvInput(graph, type, batchSize, numGroups,
                                          inputSize, {1}, name);
  return convInput.reshape({batchSize, numGroups, inputSize})
      .dimShuffle({1, 0, 2});
}

} // end namespace util
} // end namespace poplibs_test

namespace std {
std::istream &operator>>(std::istream &in, poplar::Type &type) {
  std::string token;
  in >> token;
  if (token == "half")
    type = poplar::HALF;
  else if (token == "float")
    type = poplar::FLOAT;
  else if (token == "unsigned" || token == "uint")
    type = poplar::UNSIGNED_INT;
  else if (token == "int")
    type = poplar::INT;
  else if (token == "ushort")
    type = poplar::UNSIGNED_SHORT;
  else if (token == "short")
    type = poplar::SHORT;
  else if (token == "bool")
    type = poplar::BOOL;
  else if (token == "char")
    type = poplar::CHAR;
  else if (token == "schar")
    type = poplar::SIGNED_CHAR;
  else if (token == "uchar")
    type = poplar::UNSIGNED_CHAR;
  else
    throw poputil::poplibs_error(
        "Invalid data-type <" + token +
        ">; must be half, float, uint (unsigned), int, ushort, short, char, "
        "schar, uchar or bool");
  return in;
}
} // namespace std
