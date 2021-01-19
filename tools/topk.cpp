// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>

#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>
#include <poplar/exceptions.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popnn/Loss.hpp>
#include <popnn/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poplibs_test::util;

#define FLOAT_REL_TOL 1e-6
#define HALF_REL_TOL 1e-5
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

int main(int argc, char **argv) try {

  namespace po = boost::program_options;
  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned n, k;
  unsigned batchSize = 1;
  boost::optional<unsigned> tilesPerIPU;
  Type dataType = FLOAT;
  Type indexType = UNSIGNED_INT;
  bool sortOutput = true;

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
      ("sort-output",
       po::value(&sortOutput)->default_value(sortOutput),
       "Ensure the output of the top-k is sorted")
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
  bool ignoreData = vm.count("ignore-data");
  bool showVarStorage = vm.count("show-var-storage");

  // If k was not explicitly provided, set it equal to n
  if (!vm.count("k")) {
    k = n;
  }

  std::cout << "Top-K with batch-size " << batchSize << ", input size " << n
            << ", and output size " << k << "\n";

  constexpr bool alwaysCompileCode = true;
  auto device = createTestDevice(deviceType, 1, tilesPerIPU, alwaysCompileCode);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);

  const auto in = graph.addVariable(dataType, {batchSize, n}, "in");
  // TODO: Eventually we should have an allocation function for the inputs
  // which will probably just map linearly with some kind of grain size.
  poputil::mapTensorLinearly(graph, in);

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence prog, uploadProg, downloadProg;

  Tensor outIndices;
  Tensor outValues =
      popnn::topK(graph, in, outIndices, k, sortOutput, prog, "top-k");

  std::unique_ptr<char[]> rawHostIn, rawHostIndicesOut, rawHostValuesOut;
  if (!ignoreData) {
    rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                            downloadProg, tmap);
    rawHostIndicesOut = allocateHostMemoryForTensor(
        outIndices, "outIndices", graph, uploadProg, downloadProg, tmap);
    rawHostValuesOut = allocateHostMemoryForTensor(
        outValues, "outValues", graph, uploadProg, downloadProg, tmap);
  }

  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrument", "true");
  }

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
  attachStreams(engine, tmap);

  std::vector<double> hostIn(batchSize * n);
  std::vector<unsigned> hostIndicesOut(batchSize * k);
  std::vector<double> hostValuesOut(batchSize * k);

  std::mt19937 randomEngine;
  if (!ignoreData) {
    writeRandomValues(target, dataType, hostIn, -50.0, 50.0, randomEngine);
    copy(target, hostIn, dataType, rawHostIn.get());
  }

  device.bind([&](const Device &d) { engine.loadAndRun(d); });

  bool matchesModel = true;
  if (!ignoreData) {
    copy(target, indexType, rawHostIndicesOut.get(), hostIndicesOut);
    copy(target, dataType, rawHostValuesOut.get(), hostValuesOut);
    // Verify against top-k on the host.
    std::vector<unsigned> modelIndicesOut(batchSize * k);
    std::vector<double> modelValuesOut(batchSize * k);
    {
      std::vector<unsigned> indices(n);
      for (unsigned batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](unsigned a, unsigned b) {
                            return hostIn[batchIdx * n + a] >
                                   hostIn[batchIdx * n + b];
                          });
        for (unsigned i = 0; i < k; ++i) {
          modelIndicesOut[batchIdx * k + i] = indices[i];
          modelValuesOut[batchIdx * k + i] = hostIn[batchIdx * n + indices[i]];
        }
      }
    }

    // If the output isn't already supposed to be sorted, sort it on the host
    // so that we can compare element for element to check the result.
    if (!sortOutput) {
      std::vector<unsigned> sortedIndices(k);
      std::vector<double> valBuffer(k);
      std::vector<unsigned> indexBuffer(k);
      for (unsigned batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&](unsigned a, unsigned b) {
                    return hostValuesOut[batchIdx * k + a] >
                           hostValuesOut[batchIdx * k + b];
                  });
        std::copy_n(&hostIndicesOut[batchIdx * k], k, indexBuffer.begin());
        std::copy_n(&hostValuesOut[batchIdx * k], k, valBuffer.begin());
        for (unsigned i = 0; i < k; ++i) {
          hostIndicesOut[batchIdx * k + i] = indexBuffer[sortedIndices[i]];
          hostValuesOut[batchIdx * k + i] = valBuffer[sortedIndices[i]];
        }
      }
    }

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
          indexedValues[batchIdx * k + i] = hostIn[batchIdx * n + hostIndexOut];
        }
      }
    }
    double relTolerance = dataType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
    double absTolerance = dataType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    matchesModel &= checkIsClose("indexedValues", indexedValues.data(),
                                 {batchSize, k}, modelValuesOut.data(),
                                 batchSize * k, relTolerance, absTolerance);
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
