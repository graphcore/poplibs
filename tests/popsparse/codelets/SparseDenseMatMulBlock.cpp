// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseDenseMatMulBlock
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

#include <popsolver/Model.hpp>

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "../lib/popsparse/SparseMetaInfo.hpp"
#include "SparseDenseUtils.hpp"
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;

// generate sparse block start indices
template <typename RandomEngine>
static std::vector<std::array<unsigned, 2>> generateBlockSparseIndices(
    RandomEngine &randomEngine, const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &blockSize, std::size_t n) {
  const std::vector<std::size_t> blockShape = {shape[0] / blockSize[0],
                                               shape[1] / blockSize[1]};
  // Generate n random indices that are within the flattened given shape.
  std::vector<unsigned> randomIndices(product(blockShape));
  std::iota(randomIndices.begin(), randomIndices.end(), 0);
  auto randomGen = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max - 1);
    return dist(randomEngine);
  };
  boost::range::random_shuffle(randomIndices, randomGen);
  randomIndices.resize(n);

  std::vector<std::array<unsigned, 2>> rowColumnIndices(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto unflattenedIndex =
        vectorConvert<unsigned>(unflattenIndex(blockShape, randomIndices[i]));
    rowColumnIndices[i] = {
        unflattenedIndex[0] * static_cast<unsigned>(blockSize[0]),
        unflattenedIndex[1] * static_cast<unsigned>(blockSize[1])};
  }
  return rowColumnIndices;
}

// Split the batch dimension across workers
static std::vector<unsigned int> getForwardWorkerPartition(const Target &target,
                                                           unsigned bColumns) {
  auto splits = poputil::splitRegionsBetweenWorkers(target, {{0, bColumns}}, 1);
  std::vector<unsigned int> worklist(target.getNumWorkerContexts() * 2);

  unsigned index = 0;
  for (const auto split : splits) {
    for (const auto interval : split) {
      worklist.at(index) = interval.begin();
      worklist.at(index + 1) = interval.size();
      index += 2;
    }
  }
  return worklist;
}

template <typename RandomEngine>
static std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const std::vector<std::size_t> &blockSize, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType) {

  // Factor by which row and column offsets are scaled
  const auto blockElems = product(blockSize);

  // Order indices of a by column then row
  std::sort(indices.begin(), indices.end());

  std::vector<std::vector<unsigned>> metaInfo(numBuckets);
  auto garbageDist =
      boost::random::uniform_int_distribution<unsigned>(0, 0xffff);
  std::size_t nzOffset = 0;
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    std::size_t splitIdx = 0;
    std::size_t numSplits = processedSubGroupIndices.at(bucket).size();
    for (std::size_t i = 0; i < otherSubGroupIds.size() + numSplits; ++i) {
      if (splitIdx != numSplits &&
          i == processedSubGroupIndices[bucket][splitIdx]) {

        std::vector<unsigned> rows;
        std::vector<unsigned> rowColumnCounts;
        boost::optional<unsigned> lastRowIndex;
        for (std::size_t nzIdx = nzOffset;
             nzIdx < nzOffset + subGroupNumElems[bucket].at(i); ++nzIdx) {
          if (!lastRowIndex || *lastRowIndex != indices.at(nzIdx)[0]) {
            rows.emplace_back();
            rowColumnCounts.emplace_back();
          }
          rows.back() = indices.at(nzIdx)[0];
          ++rowColumnCounts.back();
          lastRowIndex = rows.back();
        }

        metaInfo[bucket].emplace_back(processedSubGroupId);
        const auto processedSubGroupNumElems = subGroupNumElems[bucket].at(i);
        using T = unsigned;
        const auto subgroupEntryElems =
            sizeof(popsparse::BlockMetaInfo<T>::SubGroupEntry) / sizeof(T);
        const auto outputEntryElems =
            sizeof(popsparse::BlockMetaInfo<T>::OutputEntry) / sizeof(T);
        const auto totalMetaInfoElems = subgroupEntryElems +
                                        rows.size() * outputEntryElems +
                                        processedSubGroupNumElems;
        metaInfo[bucket].emplace_back(processedSubGroupNumElems * blockElems);
        metaInfo[bucket].emplace_back(totalMetaInfoElems);
        metaInfo[bucket].emplace_back(rows.size() - 1);

        // Output row -> column list meta-info
        std::vector<unsigned> outputEntryMetaInfoIndices(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) {
          const auto aRow = indices.at(nzOffset)[0];
          // First entry is offset into output memory to process.
          // bColumns are inner-most dimension.
          const auto aRowOffsetInC = aRow;
          outputEntryMetaInfoIndices[r] = metaInfo[bucket].size();
          metaInfo[bucket].push_back(aRowOffsetInC);
          metaInfo[bucket].push_back(rowColumnCounts[r] - 1);
          for (unsigned c = 0; c < rowColumnCounts[r]; ++c) {
            metaInfo[bucket].push_back(indices.at(nzOffset)[1]);
            ++nzOffset;
          }
        }
        ++splitIdx;
      } else {
        const auto otherSubGroupIdx = i - splitIdx;
        const auto subGroupId = otherSubGroupIds[otherSubGroupIdx];
        const auto numElems = subGroupNumElems[bucket][i];
        metaInfo[bucket].emplace_back(subGroupId);
        metaInfo[bucket].emplace_back(numElems * blockElems);
        // We also just use this no. of sub-elements as garbage in the meta-info
        // for the other (unprocessed) sub-groups.
        metaInfo[bucket].emplace_back(numElems + 3);
        for (std::size_t i = 0; i < numElems; ++i) {
          metaInfo[bucket].emplace_back(garbageDist(randomEngine));
        }
      }
    }
    constexpr unsigned endSubGroupId = 0;
    metaInfo[bucket].push_back(endSubGroupId);
  }
  return metaInfo;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  Type inputType = HALF;
  Type partialsType = FLOAT;
  ShapeOption<std::size_t> aShape;
  ShapeOption<std::size_t> bShape;
  ShapeOption<std::size_t> blockSize;

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
    ("block-size",
     po::value<ShapeOption<std::size_t>>(&blockSize)->required(),
     "Shape of block (rows per block, columns per block)")
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

  if (blockSize.val.size() != 2) {
    throw poplibs_error("shape of block size must be 2-dimensional");
  }

  if (aShape[0] % blockSize[0]) {
    throw poplibs_error("First dimension of a must be a multiple of the "
                        "first dimension of block size");
  }

  if (aShape[0] % blockSize[1]) {
    throw poplibs_error("Second dimension of a must be a multiple of the "
                        "second dimension of block size");
  }

  if (vertexType == VertexType::GradW && numBuckets != 1) {
    throw poplibs_error("GradW vertex can only handle --num-buckets=1");
  }

  const std::size_t modForCheck = inputType == HALF ? 4 : 2;
  if (blockSize[0] % modForCheck) {
    throw poplibs_error("First dimension of block size must be a multiple of " +
                        std::to_string(modForCheck));
  }

  if (blockSize[1] % modForCheck) {
    throw poplibs_error(
        "Second dimension of block size must be a multiple of " +
        std::to_string(modForCheck));
  }

  const std::vector<std::size_t> cShape = {aShape[0], bShape[1]};

  const auto aNumElems = product(aShape.val);
  const auto blockElems = product(blockSize.val);
  const auto aNumNonZeroElems = static_cast<std::size_t>(
      std::ceil(aNumElems / blockElems * sparsityLevel));

  if (aNumNonZeroElems / numBuckets / numSplitsPerBucket == 0) {
    throw poplibs_error("Splitting " + std::to_string(aNumNonZeroElems) +
                        " into " + std::to_string(numBuckets) + " and " +
                        std::to_string(numSplitsPerBucket) +
                        " splits leaves no "
                        "elements in some sub-groups");
  }

  const auto partialsTypeSize = partialsType == FLOAT ? 4 : 2;
  const auto inputTypeSize = inputType == FLOAT ? 4 : 2;

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popsparse::addCodelets(graph);

  std::mt19937 randomEngine;
  auto sparseIndices = generateBlockSparseIndices(
      randomEngine, aShape.val, blockSize.val, aNumNonZeroElems);
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
      randomEngine, sparseIndices, aShape.val, bShape.val, blockSize.val,
      numBuckets, processedSubGroupId, otherSubGroupIds,
      processedSubGroupIndices, subGroupNumElems, target, inputType,
      partialsType, vertexType);

  const auto metaInfoType = UNSIGNED_SHORT;

  // Allocate operands
  const auto aType = inputType;
  const auto bType = inputType;
  const auto cType = partialsType;
  std::vector<Tensor> aBuckets(numBuckets);
  std::vector<Tensor> metaInfoBuckets(numBuckets);
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    aBuckets[bucket] =
        graph.addVariable(aType, {blockElems * sum(subGroupNumElems[bucket])},
                          "aNonZero (bucket " + std::to_string(bucket) + ")");
    metaInfoBuckets[bucket] =
        graph.addVariable(metaInfoType, {hostMetaInfoBuckets[bucket].size()},
                          "metaInfo (bucket " + std::to_string(bucket) + ")");
    graph.setTileMapping(aBuckets[bucket], 0);
    graph.setTileMapping(metaInfoBuckets[bucket], 0);
  }

  // For forward, we keep a transposed view of the activations as the innermost
  // dimensions are the dimensions of a.
  const auto b = graph.addVariable(bType, {bShape.val[1], bShape.val[0]}, "b");
  const auto c = graph.addVariable(cType, {cShape[1], cShape[0]}, "c");

  graph.setTileMapping(b, 0);
  graph.setTileMapping(c, 0);

  const auto cs = graph.addComputeSet("cs0");

  std::string vertexBaseClass = "popsparse::";
  switch (vertexType) {
  case VertexType::Forward:
    vertexBaseClass += "SparseDenseMatMulBlock";
    break;
  case VertexType::Transposed:
  case VertexType::GradA:
  case VertexType::GradW:
    throw poplibs_error("Vertex type not yet supported");
    break;
  default:
    throw poplibs_error("Unrecognised vertex type");
  }
  const auto vertexClass =
      templateVertex(vertexBaseClass, inputType, partialsType, blockSize.val[0],
                     blockSize.val[1]);
  const auto v = graph.addVertex(cs, vertexClass);

  const unsigned zStrideInQ = aShape.val[0] * partialsTypeSize / 8;
  const unsigned zStrideInS = aShape.val[1] * inputTypeSize / 8;
  const unsigned maxPositiveStride = (1 << (target.getNumStrideBits() - 1)) - 1;
  if (zStrideInQ > maxPositiveStride || zStrideInS > maxPositiveStride) {
    throw poplibs_error("Strides exceed machine limits");
  }
  graph.connect(v["q"], c.flatten());
  graph.connect(v["r"], aBuckets);
  graph.connect(v["s"], b.flatten());
  graph.connect(v["metaInfo"], metaInfoBuckets);
  graph.setInitialValue(v["subGroupIdToProcess"], processedSubGroupId);
  assert(partialsType == FLOAT || c.numElements() % 2 == 0);
  const auto numPartials =
      (partialsType == FLOAT) ? c.numElements() : c.numElements() / 2;

  graph.setInitialValue(v["zeroInfo"], zeroPartials ? numPartials : 0);
  graph.setInitialValue(v["zStrideInQ"], zStrideInQ);
  graph.setInitialValue(v["zStrideInS"], zStrideInS);
  auto worklist = getForwardWorkerPartition(target, bShape.val[1]);
  auto worklistTensor = graph.addConstant(UNSIGNED_SHORT, {worklist.size()},
                                          worklist.data(), "/worklists");
  graph.setTileMapping(worklistTensor, 0);
  graph.connect(v["offsetAndNumZByWorker"], worklistTensor);
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

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  attachStreams(engine, tmap);

  std::vector<boost::multi_array<double, 1>> hostABuckets;
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    hostABuckets.emplace_back(
        boost::extents[blockElems * sum(subGroupNumElems[bucket])]);
  }
  boost::multi_array<double, 2> hostB(boost::extents[bShape[1]][bShape[0]]);
  boost::multi_array<double, 2> hostC(boost::extents[cShape[1]][cShape[0]]);

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    writeRandomValues(target, aType, hostABuckets[bucket], -1.0, 1.0,
                      randomEngine);
    copy(target, hostABuckets[bucket], aType, rawHostABuckets[bucket].get());
  }
  writeRandomValues(target, bType, hostB, -1.0, +1.00001, randomEngine);
  copy(target, hostB, bType, rawHostB.get());
  writeRandomValues(target, cType, hostC, -1.0, +1.00001, randomEngine);
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
  if (vertexType == VertexType::Forward) {
    copy(target, cType, rawHostC.get(), hostC);
  }

  if (!ignoreData) {
    // We use a dense matrix a to model this.
    boost::multi_array<double, 2> hostADense(
        boost::extents[aShape[0]][aShape[1]]);

    std::size_t nzOffset = 0;
    for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
      for (const auto &idx : processedSubGroupIndices[bucket]) {
        assert(idx < subGroupNumElems[bucket].size());
        const auto bucketNzOffset =
            blockElems * std::accumulate(subGroupNumElems[bucket].begin(),
                                         subGroupNumElems[bucket].begin() + idx,
                                         std::size_t(0));
        for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
          assert(bucketNzOffset + i < hostABuckets[bucket].num_elements());
          for (unsigned j = 0; j != blockSize.val[0]; ++j) {
            for (unsigned k = 0; k != blockSize.val[1]; ++k) {
              auto srcOffset =
                  bucketNzOffset + i * blockElems + j * blockSize.val[1] + k;
              hostADense[sparseIndices.at(nzOffset + i)[0] + j]
                        [sparseIndices.at(nzOffset + i)[1] + k] =
                            hostABuckets[bucket][srcOffset];
            }
          }
        }
        nzOffset += subGroupNumElems.at(bucket).at(idx);
      }
    }

    bool matchesModel = true;
    double relativeTolerance =
        inputType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
    double absoluteTolerance =
        inputType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    if (vertexType == VertexType::Forward) {
      boost::multi_array<double, 2> modelC(
          boost::extents[cShape[1]][cShape[0]]);
      poplibs_test::gemm::generalMatrixMultiply(hostB, hostADense, modelC,
                                                false, true);
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
