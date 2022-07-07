// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <optional>

#include <poplar/CycleCount.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>
#include <popops/Sort.hpp>
#include <popops/TopK.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poputil;

#define FLOAT_REL_TOL 1e-6
#define HALF_REL_TOL 1e-5
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

enum class API {
  Popops,
  Popnn,
  PopopsSort,
};

inline std::ostream &operator<<(std::ostream &os, const API &api) {
  switch (api) {
  case API::Popops:
    os << "popops";
    break;
  case API::Popnn:
    os << "popnn";
    break;
  case API::PopopsSort:
    os << "popops-sort";
  default:
    throw poplibs_error("Unhandled API type");
  }
  return os;
}

inline std::istream &operator>>(std::istream &is, API &api) {
  std::string token;
  is >> token;
  if (token == "popops") {
    api = API::Popops;
  } else if (token == "popnn") {
    api = API::Popnn;
  } else if (token == "popops-sort") {
    api = API::PopopsSort;
  } else {
    throw poplibs_error("Unknown API type '" + token + "'");
  }
  return is;
}

namespace popops {

inline std::istream &operator>>(std::istream &is, popops::SortOrder &o) {
  std::string token;
  is >> token;
  if (token == "none") {
    o = popops::SortOrder::NONE;
  } else if (token == "ascending") {
    o = popops::SortOrder::ASCENDING;
  } else if (token == "descending") {
    o = popops::SortOrder::DESCENDING;
  } else {
    throw poplibs_error("Unknown sort order '" + token + "'");
  }
  return is;
}

} // namespace popops

int main(int argc, char **argv) try {

  namespace po = boost::program_options;
  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned n, k;
  unsigned batchSize = 1;
  boost::optional<unsigned> tilesPerIPU;
  Type dataType = FLOAT;
  Type indexType = UNSIGNED_INT;
  bool largest = true;
  popops::SortOrder sortOrder = popops::SortOrder::ASCENDING;
  API api = API::Popops;
  bool returnIndices = true;
  bool returnValues = true;
  bool stableSort = false;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
      ("help", "Produce help message")
      ("device-type",
       po::value(&deviceType)->default_value(deviceType),
       "Device type")
      ("tiles-per-ipu",
       po::value(&tilesPerIPU),
       "Number of tiles per IPU to use")
      ("profile",
       "Output profiling report")
      ("show-var-storage",
       "When profiling, output variable liveness report also")
      ("ignore-data",
       "Don't upload/download to/from the device and consequently don't "
       "validate results")
      ("report-total-cycles",
       "Print total cycle count for the whole operation")
      ("n",
       po::value(&n)->required(),
       "Number of input elements")
      ("k",
       po::value(&k),
       "Number of output elements")
      ("batch-size",
       po::value(&batchSize)->default_value(batchSize),
       "Batch size, or number of independent top-k with k and n "
       "to compute")
      ("data-type",
       po::value(&dataType)->default_value(dataType),
       "The type of the input/output data")
      ("index-type",
       po::value(&indexType)->default_value(indexType),
       "The type of the indices")
      ("largest",
       po::value(&largest)->default_value(largest),
       "If true return the top k largest elements, otherwise return top k smallest elements")
      ("sort-order",
       po::value(&sortOrder)->default_value(sortOrder),
       "Sort order of the output of the top-k")
      ("return-indices",
       po::value(&returnIndices)->default_value(returnIndices),
       "Use API returning indices of top k values")
      ("return-values",
       po::value(&returnValues)->default_value(returnValues),
       "Use API returning top k values")
      ("stable-sort",
       po::value(&stableSort)->default_value(stableSort),
       "Maintain relative order of values that compare equal not to change in the output")
      ("api",
       po::value(&api)->default_value(api),
       "Which API to use (popops | popnn)")
      ("random-seed",
       "Use a random seed")
  ;
  // clang-format on

  // TODO: For now we assume top-k returning keys and values.
  // The eventual API could return keys, values, or both keys and values.
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  bool profile = vm.count("profile");
  bool showVarStorage = vm.count("show-var-storage");
  bool reportTotalCycles = vm.count("report-total-cycles");
  bool randomSeed = vm.count("random-seed");

  if (profile && reportTotalCycles) {
    std::cerr
        << "Can't report total cycles and profile at the same time as "
           "profiling instrumentation would skew total cycles measurement\n";
    return 1;
  }

  // If k was not explicitly provided, set it equal to n
  if (!vm.count("k")) {
    k = n;
  }

  if (!returnIndices && !returnValues) {
    std::cerr
        << "At least one of return-indices and return-values must be true\n";
    return 1;
  }

  if (stableSort && !returnIndices) {
    std::cerr << "Stable sort testing requires return-indices turned on.\n";
    return 1;
  }

  switch (api) {
  case API::Popops:
    // Nothing. Popops API supports all arguments.
    break;
  case API::Popnn:
    if (!returnIndices) {
      std::cerr << "Warning: popnn API only supports returning both indices "
                   "and values. Forcing return-indices on\n";
      returnIndices = true;
    }
    if (!returnValues) {
      std::cerr << "Warning: popnn API only supports returning both indices "
                   "and values. Forcing return-values on\n";
      returnValues = true;
    }
    if (sortOrder == popops::SortOrder::ASCENDING) {
      std::cerr << "Warning: popnn API only supports returning values sorted "
                   "in descending order. Forcing sort-order to Descending\n";
      sortOrder = popops::SortOrder::DESCENDING;
    }
    if (!largest) {
      std::cerr << "Warning: popnn API only supports returning top k largest "
                   "values. Forcing largest to true\n";
      largest = true;
    }
    if (stableSort) {
      std::cerr << "Warning: popnn API doesn't support stable sorting. Forcing "
                   "stableSort to false.\n";
      stableSort = false;
    }
    break;
  case API::PopopsSort:
    if (n != k) {
      std::cerr << "Warning: popops sort API only supports full sort. Forcing "
                   "k equal to n\n";
      k = n;
    }
    if (returnIndices && returnValues) {
      std::cerr << "Warning: popops sort API only supports returning either "
                   "keys or values not both. Forcing returnValues to false\n";
      returnValues = false;
    }
    if (stableSort) {
      std::cerr << "Warning: popops sort API doesn't support stable sorting. "
                   "Forcing stableSort to false.\n";
      stableSort = false;
    }
    break;
  }

  constexpr bool alwaysCompileCode = true;
  auto device = createTestDevice(deviceType, 1, tilesPerIPU, alwaysCompileCode);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  if (api == API::Popnn) {
    popnn::addCodelets(graph);
  }

  const std::vector<std::size_t> inShape = {batchSize, n};
  const std::vector<std::size_t> outShape = {batchSize, k};

  const auto in = graph.addVariable(dataType, inShape, "in");
  // TODO: Eventually we should have an allocation function for the inputs
  // which will probably just map linearly with some kind of grain size.
  poputil::mapTensorLinearly(graph, in);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence prog, uploadProg, downloadProg;

  Tensor outIndices;
  Tensor outValues;
  if (api == API::Popnn) {
    const bool sorted = sortOrder != popops::SortOrder::NONE;
    outValues = popnn::topK(graph, in, outIndices, k, sorted, prog, "top-k");
    // For some reason the popnn topK API leaves a singleton dimension due
    // to an implementation detail so we get a 3-dimensional tensor back
    // that we must squeeze.
    outValues = outValues.squeeze({1});
    if (returnIndices) {
      outIndices = outIndices.squeeze({1});
    }
  } else if (api == API::Popops) {
    const popops::TopKParams params(k, largest, sortOrder, stableSort);
    if (returnIndices) {
      std::tie(outValues, outIndices) =
          popops::topKWithPermutation(graph, prog, in, params, "top-k");
    } else {
      outValues = popops::topK(graph, prog, in, params, "top-k");
    }
  } else if (api == API::PopopsSort) {
    if (returnValues) {
      outValues = popops::sort(graph, in, 1, prog, "top-k");
    } else {
      std::vector<unsigned> batchIndices(n);
      std::iota(batchIndices.begin(), batchIndices.end(), 0);
      const auto iota = graph.addConstant(
          indexType, {1, n}, ArrayRef(batchIndices), "indicesInitializer");
      poputil::mapTensorLinearly(graph, iota);
      auto indices = iota.broadcast(batchSize, 0);
      outIndices = popops::sortKeyValue(graph, in, indices, 1, prog, "top-k");
    }
  }

  // Check types/shapes returned by the API
  if (returnIndices) {
    if (outIndices.elementType() != indexType) {
      std::cerr << "Actual index type (" << outIndices.elementType()
                << ") is not the requested index type ( " << indexType << ")\n";
      return 1;
    }
    if (outIndices.shape() != outShape) {
      std::cerr << "Shape of returned indices (" << outIndices.shape()
                << ") does not match the expected shape (" << outShape << ")\n";
      return 1;
    }
  }
  if (returnValues) {
    if (outValues.elementType() != dataType) {
      std::cerr << "Actual value type (" << outIndices.elementType()
                << ") is not the requested value type ( " << dataType << ")\n";
    }
    if (outValues.shape() != outShape) {
      std::cerr << "Shape of returned values (" << outValues.shape()
                << ") does not match the expected shape (" << outShape << ")\n";
      return 1;
    }
  }

  std::unique_ptr<char[]> rawHostIn, rawHostIndicesOut, rawHostValuesOut;
  rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                          downloadProg, tmap);
  if (returnIndices) {
    rawHostIndicesOut = allocateHostMemoryForTensor(
        outIndices, "outIndices", graph, uploadProg, downloadProg, tmap);
  }
  if (returnValues) {
    rawHostValuesOut = allocateHostMemoryForTensor(
        outValues, "outValues", graph, uploadProg, downloadProg, tmap);
  }

  std::optional<TempDir> tempDir;
  OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }

  Tensor cycleCounter;
  if (reportTotalCycles) {
    cycleCounter =
        cycleCount(graph, prog, 0, SyncType::INTERNAL, "measure-total-cycles");
    graph.createHostRead("totalCycleCount", cycleCounter);
  }

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);

  std::vector<double> hostIn(batchSize * n);
  std::vector<unsigned> hostIndicesOut(batchSize * k);
  std::vector<double> hostValuesOut(batchSize * k);

  std::mt19937 randomEngine;
  if (randomSeed) {
    const auto seed = std::random_device{}();
    std::cout << "Seeding random engine with seed " << seed << "\n";
    randomEngine.seed(seed);
  }
  // TODO: Check what happens with NaN values..
  double rangeMin = (dataType == UNSIGNED_INT) ? 0 : -50.0;
  double rangeMax = 50.0;
  if (stableSort) {
    const double repetitionProbability = 0.85;
    writeRandomValuesWithRepetitions(target, dataType, hostIn, rangeMin,
                                     rangeMax, repetitionProbability,
                                     randomEngine);
  } else {
    writeRandomValues(target, dataType, hostIn, rangeMin, rangeMax,
                      randomEngine);
  }
  copy(target, hostIn, dataType, rawHostIn.get());

  device.bind([&](const Device &d) {
    engine.loadAndRun(d);
    if (reportTotalCycles) {
      std::uint64_t cycleCount;
      engine.readTensor("totalCycleCount", &cycleCount, &cycleCount + 1);
      std::cout << "Total cycles for top-k program were " << cycleCount << "\n";
    }
  });

  bool matchesModel = true;
  if (returnIndices) {
    copy(target, indexType, rawHostIndicesOut.get(), hostIndicesOut);
  }
  if (returnValues) {
    copy(target, dataType, rawHostValuesOut.get(), hostValuesOut);
  }

  // Verify against top-k on the host.
  //
  // Used in partial_sort to bring largest or smallest k to the front
  const auto topKComparator = [&]() -> std::function<bool(double, double)> {
    if (largest) {
      return std::greater<double>{};
    } else {
      return std::less<double>{};
    }
  }();

  // Used to sort largest/smallest k values after partial_sort
  const auto sortComparator = [&]() -> std::function<bool(double, double)> {
    if (sortOrder == popops::SortOrder::DESCENDING) {
      return std::greater<double>{};
    } else {
      // Also if SortOrder::NONE to make validation easier
      return std::less<double>{};
    }
  }();

  std::vector<unsigned> modelIndicesOut(batchSize * k);
  std::vector<double> modelValuesOut(batchSize * k);
  {
    std::vector<unsigned> indices(n);
    for (unsigned batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
      std::iota(indices.begin(), indices.end(), 0);
      if (k != n) {
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](unsigned a, unsigned b) {
                            const auto &aKey = hostIn[batchIdx * n + a];
                            const auto &bKey = hostIn[batchIdx * n + b];
                            if (aKey != bKey) {
                              return topKComparator(aKey, bKey);
                            }
                            return a < b;
                          });
      }
      std::stable_sort(indices.begin(), indices.begin() + k,
                       [&](unsigned a, unsigned b) {
                         const auto &aKey = hostIn[batchIdx * n + a];
                         const auto &bKey = hostIn[batchIdx * n + b];
                         return sortComparator(aKey, bKey);
                       });
      for (unsigned i = 0; i < k; ++i) {
        modelIndicesOut[batchIdx * k + i] = indices[i];
        modelValuesOut[batchIdx * k + i] = hostIn[batchIdx * n + indices[i]];
      }
    }
  }

  // If the output isn't already supposed to be sorted, sort it on the host
  // so that we can compare element for element to check the result.
  if (sortOrder == popops::SortOrder::NONE) {
    std::vector<unsigned> sortedIndices(k);
    std::vector<double> valBuffer(k);
    std::vector<unsigned> indexBuffer(k);
    for (unsigned batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
      std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
      std::stable_sort(sortedIndices.begin(), sortedIndices.end(),
                       [&](unsigned a, unsigned b) {
                         return sortComparator(hostValuesOut[batchIdx * k + a],
                                               hostValuesOut[batchIdx * k + b]);
                       });
      if (returnIndices) {
        std::copy_n(&hostIndicesOut[batchIdx * k], k, indexBuffer.begin());
        for (unsigned i = 0; i < k; ++i) {
          hostIndicesOut[batchIdx * k + i] = indexBuffer[sortedIndices[i]];
        }
      }
      if (returnValues) {
        std::copy_n(&hostValuesOut[batchIdx * k], k, valBuffer.begin());
        for (unsigned i = 0; i < k; ++i) {
          hostValuesOut[batchIdx * k + i] = valBuffer[sortedIndices[i]];
        }
      }
    }
  }

  double relTolerance = dataType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  double absTolerance = dataType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
  if (returnIndices) {
    if (!stableSort) {
      // Because 2 values might be equal and therefore the order of the indices
      // is not well defined, we don't directly check the indices but instead
      // check the values the indices point to match the data.
      std::vector<double> indexedValues(batchSize * k);
      for (unsigned batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        for (unsigned i = 0; i < k; ++i) {
          const auto hostIndexOut = hostIndicesOut[batchIdx * k + i];
          if (hostIndexOut >= n) {
            std::cerr << "indices[" << batchIdx << "][" << i
                      << "]=" << hostIndexOut
                      << " which is not a valid index (n=" << n << ")\n";
            matchesModel = false;
          } else {
            indexedValues[batchIdx * k + i] =
                hostIn[batchIdx * n + hostIndexOut];
          }
        }
      }
      matchesModel &= checkIsClose("indexedValues", indexedValues.data(),
                                   {batchSize, k}, modelValuesOut.data(),
                                   batchSize * k, relTolerance, absTolerance);
    } else {
      // NOTE: If sortOrder == SortOrder::NONE, we did a stable sort of
      // the resulting data so this test should also work.
      matchesModel &=
          checkEqual("indices", hostIndicesOut.data(), {batchSize, k},
                     modelIndicesOut.data(), batchSize * k);
    }
  }
  if (returnValues) {
    matchesModel &= checkIsClose("values", hostValuesOut.data(), {batchSize, k},
                                 modelValuesOut.data(), batchSize * k,
                                 relTolerance, absTolerance);
  }

  if (profile) {
    OptionFlags reportOptions{{"showExecutionSteps", "true"}};
    if (showVarStorage) {
      reportOptions.set("showVarStorage", "true");
    }
    engine.printProfileSummary(std::cout, std::move(reportOptions));
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
} catch (const graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;
  return 77;
}
