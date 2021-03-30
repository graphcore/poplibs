// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseDenseMatMulElementWise
#include <poplibs_support/TestDevice.hpp>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#include <iostream>
#include <random>
#include <vector>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <poplar/Graph.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/print.hpp>

#include <popsparse/codelets.hpp>

#include "SparseDensePartitionElementWise.hpp"
#include "SparseDenseUtils.hpp"

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  Type inputType = HALF;
  Type partialsType = FLOAT;
  ShapeOption<std::size_t> aShape;
  ShapeOption<std::size_t> bShape;
  double sparsityLevel = 0.1;
  unsigned numOtherSubGroups = 5;
  unsigned numOtherSubGroupElems = 30;
  unsigned numBuckets = 1;
  unsigned numSplitsPerBucket = 1;
  VertexType vertexType = VertexType::Forward;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling information for the program")
    ("ignore-data", "Don't validate outputs, don't add streams etc."
     " Useful for profiling")
    ("show-execution-steps", "If profiling, show execution steps in the "
     "summary")
    ("show-var-storage", "If profiling, show variable liveness information in "
     " the summary")
    ("input-type",
     po::value<Type>(&inputType)->default_value(inputType),
     "Input type")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(partialsType),
     "Partials type")
    ("a-shape",
     po::value<ShapeOption<std::size_t>>(&aShape)->required(),
     "Shape of A ")
    ("b-shape",
     po::value<ShapeOption<std::size_t>>(&bShape)->required(),
     "Shape of B (columns must be multiples if 4 for half and multiples of 2 "
     "for float")
    ("sparsity-level",
     po::value<double>(&sparsityLevel)->default_value(sparsityLevel),
     "Level of sparsity of operand A")
    ("num-other-sub-groups",
     po::value<unsigned>(&numOtherSubGroups)->default_value(numOtherSubGroups),
     "Number of other (unprocessed) sub-groups to include in meta-info")
    ("num-other-sub-group-elements",
     po::value<unsigned>(&numOtherSubGroupElems)->
                default_value(numOtherSubGroupElems),
     "Number of elements in meta-info for other sub-groups (unprocessed)")
    ("num-buckets",
     po::value<unsigned>(&numBuckets)->default_value(numBuckets),
     "Number of buckets to generate and give to the codelet. Each bucket "
     "has same number other sub-groups, and same number of other sub-group "
     "elements. Number of non-zero elements to actually process is spread "
     "between buckets")
    ("num-splits-per-bucket",
     po::value<unsigned>(&numSplitsPerBucket)->default_value(numSplitsPerBucket),
     "How many times to split the processed sub-group in each bucket. It is "
     "valid to get the same sub-group multiple times in a bucket and this "
     "allows testing.")
    ("zero-partials", "Whether or not to zero partials. Default is to not "
     "do so")
    ("vertex-type",
     po::value<VertexType>(&vertexType)->default_value(vertexType),
     "Which vertex to test (Forward | GradA | Transposed | GradW)")
  ;

  // GradWAmp vertex not supported for Element sparsity
  if (vertexType == VertexType::GradWAmp) {
    throw poplibs_error("GradWAmp is not supported for element sparsity");
  }

  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  bool profile = vm.count("profile");
  bool ignoreData = vm.count("ignore-data");
  bool showExecutionSteps = vm.count("show-execution-steps");
  bool showVarStorage = vm.count("show-var-storage");
  bool zeroPartials = vm.count("zero-partials");

  if (sparsityLevel <= 0 || sparsityLevel >= 1) {
    throw poplibs_error("sparsity-level must be in range (0, 1) but " +
                        std::to_string(sparsityLevel) + " was given");
  }

  if (aShape.val.size() != 2) {
    throw poplibs_error("shape of a must be 2-dimensional");
  }

  if (bShape.val.size() != 2) {
    throw poplibs_error("shape of b must be 2-dimensional");
  }

  if (aShape[1] != bShape[0]) {
    throw poplibs_error("size of inner dimension of a (" +
                        std::to_string(aShape[1]) +
                        ") must match outer dimension of b (" +
                        std::to_string(bShape[0]) + ")");
  }

  if (vertexType == VertexType::GradW && numBuckets != 1) {
    throw poplibs_error("GradW vertex can only handle --num-buckets=1");
  }

  std::size_t modForCheck = inputType == HALF ? 4 : 2;

  if (bShape[1] % modForCheck) {
    throw poplibs_error("sizes of second dimension of b must be multiple of " +
                        std::to_string(modForCheck));
  }

  const std::vector<std::size_t> cShape = {aShape[0], bShape[1]};

  const auto aNumElems = product(aShape.val);
  const auto aNumNonZeroElems =
      static_cast<std::size_t>(std::ceil(aNumElems * sparsityLevel));

  if (aNumNonZeroElems / numBuckets / numSplitsPerBucket == 0) {
    throw poplibs_error("Splitting " + std::to_string(aNumNonZeroElems) +
                        " into " + std::to_string(numBuckets) + " and " +
                        std::to_string(numSplitsPerBucket) +
                        " splits leaves no "
                        "elements in some sub-groups");
  }

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popsparse::addCodelets(graph);

  std::mt19937 randomEngine;
  auto sparseIndices =
      generateSparseIndices(randomEngine, aShape.val, aNumNonZeroElems);

  unsigned processedSubGroupId;
  std::vector<unsigned> otherSubGroupIds;
  std::tie(processedSubGroupId, otherSubGroupIds) =
      generateSparseSubGroupIds(randomEngine, 1 + numOtherSubGroups, 1, 1000);
  std::vector<std::vector<unsigned>> processedSubGroupIndices;
  std::vector<std::vector<unsigned>> subGroupNumElems;
  std::tie(processedSubGroupIndices, subGroupNumElems) = partitionSubGroupElems(
      randomEngine, sparseIndices.size(), numBuckets, numSplitsPerBucket,
      numOtherSubGroups, numOtherSubGroupElems);
  const auto hostMetaInfoBuckets = generateMetaInfoAndPartition(
      randomEngine, sparseIndices, aShape.val, bShape.val, numBuckets,
      processedSubGroupId, otherSubGroupIds, processedSubGroupIndices,
      subGroupNumElems, target, inputType, partialsType, vertexType);

  // TODO: Check values in meta-info to ensure they are representable by this
  // type.
  const auto metaInfoType = UNSIGNED_SHORT;

  // Allocate operands
  const auto aType = vertexType == VertexType::GradW ? partialsType : inputType;
  const auto bType =
      vertexType == VertexType::Transposed ? partialsType : inputType;
  const auto cType =
      vertexType == VertexType::Forward || vertexType == VertexType::GradA
          ? partialsType
          : inputType;
  std::vector<Tensor> aBuckets(numBuckets);
  std::vector<Tensor> metaInfoBuckets(numBuckets);
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    aBuckets[bucket] =
        graph.addVariable(aType, {sum(subGroupNumElems[bucket])},
                          "aNonZero (bucket " + std::to_string(bucket) + ")");
    metaInfoBuckets[bucket] =
        graph.addVariable(metaInfoType, {hostMetaInfoBuckets[bucket].size()},
                          "metaInfo (bucket " + std::to_string(bucket) + ")");
    graph.setTileMapping(aBuckets[bucket], 0);
    graph.setTileMapping(metaInfoBuckets[bucket], 0);
  }
  const auto b = graph.addVariable(bType, bShape.val, "b");
  const auto c = graph.addVariable(cType, cShape, "c");

  graph.setTileMapping(b, 0);
  graph.setTileMapping(c, 0);

  const auto cs = graph.addComputeSet("cs0");

  std::string vertexBaseClass = "popsparse::";
  switch (vertexType) {
  case VertexType::Forward:
    vertexBaseClass += "SparseDenseMatMulElementWise";
    break;
  case VertexType::Transposed:
    vertexBaseClass += "SparseDenseMatMulElementWiseTranspose";
    break;
  case VertexType::GradA:
    vertexBaseClass += "SparseDenseMatMulGradAElementWise";
    break;
  case VertexType::GradW:
    vertexBaseClass += "SparseDenseMatMulGradWElementWise";
    break;
  default:
    throw poplibs_error("Unrecognised vertex type");
  }
  const auto vertexClass =
      templateVertex(vertexBaseClass, inputType, partialsType);
  const auto v = graph.addVertex(cs, vertexClass);

  const auto getZeroInfo = [&](const std::size_t numElems,
                               const Type &dataType) {
    assert(8 % target.getTypeSize(dataType) == 0);
    const auto elemsPer64Bits = 8 / target.getTypeSize(dataType);
    // NOTE: We don't properly enforce this requirement for a multiple of
    // 64-bits for the nz values in the buckets hence this assert will
    // fire if it happens to be wrong.
    assert(numElems % elemsPer64Bits == 0);
    return numElems / elemsPer64Bits;
  };

  if (vertexType == VertexType::GradW) {
    graph.connect(v["qGrad"], c.flatten());
    graph.connect(v["rGrad"], aBuckets.at(0));
    graph.connect(v["s"], b.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets.at(0));
    const auto deviceProcessedSubGroupId =
        graph.addConstant(metaInfoType, {}, processedSubGroupId);
    graph.setTileMapping(deviceProcessedSubGroupId, 0);
    graph.connect(v["subGroupIdToProcess"], deviceProcessedSubGroupId);
    graph.setInitialValue(v["numZ"], bShape[1]);
    graph.setInitialValue(
        v["zeroInfo"],
        zeroPartials ? getZeroInfo(aBuckets.at(0).numElements(), partialsType)
                     : 0);
  } else {
    graph.connect(v["q"], c.flatten());
    graph.connect(v["r"], aBuckets);
    graph.connect(v["s"], b.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets);
    graph.setInitialValue(v["subGroupIdToProcess"], processedSubGroupId);
    const auto numPartials = vertexType == VertexType::Transposed
                                 ? b.numElements()
                                 : c.numElements();
    graph.setInitialValue(v["zeroInfo"],
                          zeroPartials ? getZeroInfo(numPartials, partialsType)
                                       : 0);
  }
  graph.setTileMapping(v, 0);

  Sequence prog;
  prog.add(Execute(cs));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostB, rawHostC;
  std::vector<std::unique_ptr<char[]>> rawHostABuckets(numBuckets),
      rawHostMetaInfoBuckets(numBuckets);
  if (!ignoreData) {
    for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
      rawHostABuckets[bucket] = allocateHostMemoryForTensor(
          aBuckets[bucket], "a[" + std::to_string(bucket) + "]", graph,
          uploadProg, downloadProg, tmap);
      rawHostMetaInfoBuckets[bucket] = allocateHostMemoryForTensor(
          metaInfoBuckets[bucket], "metaInfo[" + std::to_string(bucket) + "]",
          graph, uploadProg, downloadProg, tmap);
    }
    rawHostB = allocateHostMemoryForTensor(b, "b", graph, uploadProg,
                                           downloadProg, tmap);
    rawHostC = allocateHostMemoryForTensor(c, "c", graph, uploadProg,
                                           downloadProg, tmap);
  }

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg});
  attachStreams(engine, tmap);

  std::vector<boost::multi_array<double, 1>> hostABuckets;
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    hostABuckets.emplace_back(boost::extents[sum(subGroupNumElems[bucket])]);
  }
  boost::multi_array<double, 2> hostB(boost::extents[bShape[0]][bShape[1]]);
  boost::multi_array<double, 2> hostC(boost::extents[cShape[0]][cShape[1]]);

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    writeRandomValues(target, aType, hostABuckets[bucket], -1.0, +1.0,
                      randomEngine);
    copy(target, hostABuckets[bucket], aType, rawHostABuckets[bucket].get());
  }
  writeRandomValues(target, bType, hostB, -1.0, +1.0, randomEngine);
  copy(target, hostB, bType, rawHostB.get());
  writeRandomValues(target, cType, hostC, -1.0, +1.0, randomEngine);
  copy(target, hostC, cType, rawHostC.get());
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    copy(target, hostMetaInfoBuckets[bucket], metaInfoType,
         rawHostMetaInfoBuckets[bucket].get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.setPrintStream(std::cerr);
    engine.run();
  });

  // Store initial values of operands before pulling partials from device
  // as these are needed to test partials zeroing (or not zeroing).
  // NOTE: Need deep copy hence type is explicit to avoid getting a view
  const boost::multi_array<double, 1> origA = hostABuckets.at(0);
  const boost::multi_array<double, 2> origB = hostB;
  const boost::multi_array<double, 2> origC = hostC;
  if (vertexType == VertexType::GradW) {
    copy(target, aType, rawHostABuckets.at(0).get(), hostABuckets.at(0));
  } else if (vertexType == VertexType::Transposed) {
    copy(target, bType, rawHostB.get(), hostB);
  } else if (vertexType == VertexType::Forward ||
             vertexType == VertexType::GradA) {
    copy(target, cType, rawHostC.get(), hostC);
  }

  if (!ignoreData) {
    // We use a dense matrix a to model this.
    boost::multi_array<double, 2> hostADense(
        boost::extents[aShape[0]][aShape[1]]);

    if (vertexType != VertexType::GradW) {
      std::size_t nzOffset = 0;
      for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
        for (const auto &idx : processedSubGroupIndices[bucket]) {
          assert(idx < subGroupNumElems[bucket].size());
          const auto bucketNzOffset = std::accumulate(
              subGroupNumElems[bucket].begin(),
              subGroupNumElems[bucket].begin() + idx, std::size_t(0));
          for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
            assert(bucketNzOffset + i < hostABuckets[bucket].num_elements());
            hostADense[sparseIndices.at(nzOffset + i)[0]]
                      [sparseIndices.at(nzOffset + i)[1]] =
                          hostABuckets[bucket][bucketNzOffset + i];
          }
          nzOffset += subGroupNumElems.at(bucket).at(idx);
        }
      }
    }
    bool matchesModel = true;
    double relativeTolerance =
        inputType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
    double absoluteTolerance =
        inputType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    if (vertexType == VertexType::GradW) {
      boost::multi_array<double, 2> modelADense(
          boost::extents[aShape[0]][aShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostC, hostB, modelADense,
                                                false, true);

      // Now get the model sparse a, we do this to see if the actual sparse a
      // overwrote any of the other positions that weren't part of the
      // processed sub-group.
      boost::multi_array<double, 1> modelA(
          boost::extents[sum(subGroupNumElems.at(0))]);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelA.num_elements(); ++i) {
          modelA.data()[i] = origA.data()[i];
        }
      }
      std::size_t nzOffset = 0;
      for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
        for (const auto &idx : processedSubGroupIndices[bucket]) {
          const auto bucketNzOffset = std::accumulate(
              subGroupNumElems[bucket].begin(),
              subGroupNumElems[bucket].begin() + idx, std::size_t(0));
          for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
            modelA[bucketNzOffset + i] +=
                modelADense[sparseIndices.at(nzOffset + i)[0]]
                           [sparseIndices.at(nzOffset + i)[1]];
          }
          nzOffset += subGroupNumElems[bucket][idx];
        }
      }

      matchesModel = checkIsClose("modelA", hostABuckets.at(0), modelA,
                                  relativeTolerance, absoluteTolerance);
    } else if (vertexType == VertexType::Transposed) {
      boost::multi_array<double, 2> modelB(
          boost::extents[bShape[0]][bShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostADense, hostC, modelB, true,
                                                false);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelB.num_elements(); ++i) {
          modelB.data()[i] += origB.data()[i];
        }
      }
      matchesModel = checkIsClose("modelB", hostB, modelB, relativeTolerance,
                                  absoluteTolerance);
    } else if (vertexType == VertexType::Forward ||
               vertexType == VertexType::GradA) {
      boost::multi_array<double, 2> modelC(
          boost::extents[cShape[0]][cShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostADense, hostB, modelC,
                                                false, false);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelC.num_elements(); ++i) {
          modelC.data()[i] += origC.data()[i];
        }
      }
      matchesModel = checkIsClose("modelC", hostC, modelC, relativeTolerance,
                                  absoluteTolerance);
    } else {
      throw poputil::poplibs_error("Unhandled vertex type");
    }

    if (!matchesModel) {
      std::cerr << "Validation failed\n";
      return 1;
    }
  }

  if (profile) {
    engine.printProfileSummary(
        std::cerr,
        {{"showExecutionSteps", (showExecutionSteps ? "true" : "false")},
         {"showVarStorage", (showVarStorage ? "true" : "false")}});
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
