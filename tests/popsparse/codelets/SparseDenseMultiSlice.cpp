// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseDenseMultiSlice
#include <poplibs_support/TestDevice.hpp>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#include <iostream>
#include <optional>
#include <random>
#include <vector>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <poplar/Graph.hpp>

#include "poplibs_test/TempDir.hpp"
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/print.hpp>

#include <popsparse/codelets.hpp>

#include "../lib/popsparse/SparseCodeletMetaInfoScale.hpp"
#include "SparseDensePartitionBlock.hpp"
#include "SparseDensePartitionElementWise.hpp"
#include "SparseDenseUtils.hpp"

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poputil;
using namespace poplibs_support;

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  Type inputType = HALF;
  ShapeOption<std::size_t> baseTShape;
  auto blockSize = ShapeOption<std::size_t>(1);

  unsigned numOffsets;
  double sparsityLevel = 0.1;
  double scale = 0.5;
  unsigned numOtherSubGroups = 5;
  unsigned numOtherSubGroupElems = 30;
  unsigned numBuckets = 1;
  unsigned numSplitsPerBucket = 1;
  unsigned rowOffset = 0;
  unsigned zSize = 4;
  bool initialiseSubT = false;
  bool updateAdd = false;
  bool debugPrint = false;
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
    ("update-add",
     po::value<bool>(&updateAdd)->default_value(updateAdd),
     "Test the update add vertex")
    ("scale",
     po::value<double>(&scale)->default_value(scale),
     "Scale to use when testing the update add vertex")
    ("baseT-shape",
     po::value<ShapeOption<std::size_t>>(&baseTShape)->required(),
     "Shape of baseT input tensor ")
    ("block-size",
     po::value<ShapeOption<std::size_t>>(&blockSize)->default_value(blockSize),
     "Shape of block (rows per block, columns per block). Default(1,1) denotes"
     " elementwise sparsity")
    ("row-offset",
      po::value<unsigned>(&rowOffset)->default_value(rowOffset),
      "Row offset - attatch rows from row-offet onward to the vertex")
    ("z-size",
      po::value<unsigned>(&zSize)->default_value(zSize),
      "The size of the z dimension used in mat-mul sparse data generation")
    ("offsets",
     po::value<unsigned>(&numOffsets)->required(),
     "Size of the offsets tensor: The number of rows to extract")
    ("initialise-subT",
     po::value<bool>(&initialiseSubT)->default_value(initialiseSubT),
    "The vertex is required to update so initialise subT with non zero values")
    ("sparsity-level",
     po::value<double>(&sparsityLevel)->default_value(sparsityLevel),
     "Level of sparsity of baseT")
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
    ("debug-print",
     po::value<bool>(&debugPrint)->default_value(debugPrint),
    "Print inputs, results to aid with debugging")
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
  if (numOtherSubGroups == 0) {
    numOtherSubGroupElems = 0;
  }

  bool profile = vm.count("profile");
  bool ignoreData = vm.count("ignore-data");
  bool showExecutionSteps = vm.count("show-execution-steps");
  bool showVarStorage = vm.count("show-var-storage");
  bool doElementWiseTest =
      std::all_of(blockSize.val.begin(), blockSize.val.end(),
                  [](std::size_t dim) { return dim == 1; });
  if (numOtherSubGroups == 0) {
    numOtherSubGroupElems = 0;
  }

  if (blockSize.val.size() == 1) {
    blockSize.val.push_back(blockSize[0]);
  }

  if (sparsityLevel <= 0 || sparsityLevel >= 1) {
    throw poplibs_error("sparsity-level must be in range (0, 1) but " +
                        std::to_string(sparsityLevel) + " was given");
  }

  if (baseTShape.val.size() != 2) {
    throw poplibs_error("shape of baseT must be 2-dimensional");
  }
  if (rowOffset >= baseTShape[0]) {
    throw poplibs_error("Row offset cannot be greater than rows in baseT");
  }
  if (inputType == HALF && (baseTShape[1] % 2) && !updateAdd) {
    throw poplibs_error("Slice vertex with data type half only supports an"
                        " even number of columns in baseT.");
  }
  const std::vector<std::size_t> subTShape = {numOffsets, baseTShape[1]};
  // With row offset the populated part of the baseT tensor is only this large,
  // generate sparse data to populate that piece
  std::vector<std::size_t> offsetBaseTShape = {baseTShape[0] - rowOffset,
                                               baseTShape[1]};

  const auto baseTNumElems = product(offsetBaseTShape);
  const auto blockElems = product(blockSize.val);
  const auto baseTNumNonZeroElems = static_cast<std::size_t>(
      std::ceil(baseTNumElems / blockElems * sparsityLevel));

  if (baseTNumNonZeroElems / numBuckets / numSplitsPerBucket == 0) {
    throw poplibs_error("Splitting " + std::to_string(baseTNumNonZeroElems) +
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
  auto sparseIndices = [&]() {
    if (doElementWiseTest) {
      return generateSparseIndices(randomEngine, offsetBaseTShape,
                                   baseTNumNonZeroElems);
    } else {
      return generateBlockSparseIndices(randomEngine, offsetBaseTShape,
                                        blockSize.val, baseTNumNonZeroElems);
    }
  }();

  unsigned processedSubGroupId;
  std::vector<unsigned> otherSubGroupIds;
  std::tie(processedSubGroupId, otherSubGroupIds) =
      generateSparseSubGroupIds(randomEngine, 1 + numOtherSubGroups, 1, 1000);
  // An arbritrary value which the vertex under test has to compare to and
  // match before processing the subGroup data.  This really means "which
  // partition of the input columns does the metadata hold, and the vertex
  // need to update"
  const unsigned yPartitionToProcess = 17;
  // Use 1 row per partition as the rowOffset may not be divisible by anything
  const unsigned rowsPerPartition = 1;

  std::vector<std::vector<unsigned>> processedSubGroupIndices;
  std::vector<std::vector<unsigned>> subGroupNumElems;
  std::tie(processedSubGroupIndices, subGroupNumElems) = partitionSubGroupElems(
      randomEngine, sparseIndices.size(), numBuckets, numSplitsPerBucket,
      numOtherSubGroups, numOtherSubGroupElems);

  const auto hostMetaInfoBuckets = [&]() {
    if (doElementWiseTest) {
      return generateMetaInfoAndPartition(
          randomEngine, sparseIndices, offsetBaseTShape,
          {offsetBaseTShape[1], zSize}, numBuckets, processedSubGroupId,
          otherSubGroupIds, processedSubGroupIndices, subGroupNumElems, target,
          inputType, inputType, VertexType::GradW, rowOffset,
          yPartitionToProcess);
    } else {
      return generateMetaInfoAndPartition(
          randomEngine, sparseIndices, offsetBaseTShape,
          {offsetBaseTShape[1], zSize}, blockSize.val, numBuckets,
          processedSubGroupId, otherSubGroupIds, processedSubGroupIndices,
          subGroupNumElems, target, inputType, FLOAT, VertexType::GradW,
          rowOffset, yPartitionToProcess);
    }
  }();

  // Check values in meta-info to ensure they are representable by this type
  const auto metaInfoType = UNSIGNED_SHORT;
  for (unsigned i = 0; i < hostMetaInfoBuckets.size(); i++) {
    if (std::any_of(hostMetaInfoBuckets[i].begin(),
                    hostMetaInfoBuckets[i].end(), [](const unsigned a) {
                      return a > std::numeric_limits<unsigned short>::max();
                    })) {
      throw poplibs_error("Meta Data exceeds type size.");
    }
  }

  // Allocate operands

  // When the input type is HALF, Update add will write to an Nz "partial" of
  // type FLOAT, giving the possibility of maintaining greater resolution
  // when accumulating multiple updates.
  const auto nzType = (updateAdd && inputType == HALF) ? FLOAT : inputType;

  std::vector<Tensor> nzBuckets(numBuckets);
  std::vector<Tensor> metaInfoBuckets(numBuckets);
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    nzBuckets[bucket] =
        graph.addVariable(nzType, {blockElems * sum(subGroupNumElems[bucket])},
                          "NonZero (bucket " + std::to_string(bucket) + ")");
    metaInfoBuckets[bucket] =
        graph.addVariable(metaInfoType, {hostMetaInfoBuckets[bucket].size()},
                          "metaInfo (bucket " + std::to_string(bucket) + ")");
    graph.setTileMapping(nzBuckets[bucket], 0);
    graph.setTileMapping(metaInfoBuckets[bucket], 0);
  }
  const auto offsets = graph.addVariable(UNSIGNED_INT, {numOffsets}, "offsets");
  const auto subT = graph.addVariable(inputType, subTShape, "subT");

  graph.setTileMapping(offsets, 0);
  graph.setTileMapping(subT, 0);

  const auto cs = graph.addComputeSet("cs0");

  std::string vertexBaseClass =
      doElementWiseTest
          ? (updateAdd ? "popsparse::SparseDenseMultiUpdateAddElementWise"
                       : "popsparse::SparseDenseMultiSliceElementWise")
          : (updateAdd ? "popsparse::SparseDenseMultiUpdateAddBlock"
                       : "popsparse::SparseDenseMultiSliceBlock");

  const bool vectorise =
      (blockSize.val[1] % target.getVectorWidth(nzType)) == 0;

  auto bytesPerBlockRow = target.getTypeSize(nzType) * blockSize.val[1];
  const unsigned vectorWidthInBytes =
      (bytesPerBlockRow % 8 == 0) ? 8 : ((bytesPerBlockRow % 4 == 0) ? 4 : 2);

  const auto vertexClass =
      doElementWiseTest ? templateVertex(vertexBaseClass, inputType)
      : updateAdd
          ? templateVertex(vertexBaseClass, inputType, vectorise)
          : templateVertex(vertexBaseClass, inputType, vectorWidthInBytes);
  const auto v = graph.addVertex(cs, vertexClass);

  graph.setInitialValue(v["subColumns"], baseTShape[1]);

  graph.connect(v["offsets"], offsets);
  graph.connect(v["baseTNZ"], nzBuckets);
  graph.connect(v["baseTMetaInfo"], metaInfoBuckets);
  graph.connect(v["subT"], subT.flatten());
  graph.setInitialValue(v["numOffsets"], numOffsets);
  graph.setInitialValue(v["rowsPerPartition"], rowsPerPartition);

  if (doElementWiseTest) {
    graph.setInitialValue(v["nzScaleFactor"], reciprocalMulFactor(zSize));
  } else {
    graph.setInitialValue(v["blockRows"], blockSize.val[0]);
    graph.setInitialValue(v["blockColumns"], blockSize.val[1]);
  }
  if (updateAdd) {
    // Connect tensor for scale and yPartitionToProcess.  Due to the the use
    // case where dense data is exchanged, yPartitionToProcess is a tensor for
    // the Update vertex, but in the vertex state for the Slice vertex
    auto scaleT = graph.addConstant(FLOAT, {}, scale, "Scale");
    graph.setTileMapping(scaleT, 0);
    graph.connect(v["scale"], scaleT);

    auto yPartitionToProcessT = graph.addConstant(
        UNSIGNED_INT, {}, yPartitionToProcess, "yPartitionToProcess");
    graph.setTileMapping(yPartitionToProcessT, 0);

    graph.connect(v["yPartitionToProcess"], yPartitionToProcessT);
  } else {
    graph.setInitialValue(v["yPartitionToProcess"], yPartitionToProcess);
  }

  graph.setTileMapping(v, 0);

  Sequence prog;
  prog.add(Execute(cs));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostOffsets, rawHostSubT;
  std::vector<std::unique_ptr<char[]>> rawHostNZBuckets(numBuckets),
      rawHostMetaInfoBuckets(numBuckets);

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    rawHostNZBuckets[bucket] = allocateHostMemoryForTensor(
        nzBuckets[bucket], "a[" + std::to_string(bucket) + "]", graph,
        uploadProg, downloadProg, tmap);
    rawHostMetaInfoBuckets[bucket] = allocateHostMemoryForTensor(
        metaInfoBuckets[bucket], "metaInfo[" + std::to_string(bucket) + "]",
        graph, uploadProg, downloadProg, tmap);
  }
  rawHostOffsets = allocateHostMemoryForTensor(offsets, "offsets", graph,
                                               uploadProg, downloadProg, tmap);
  rawHostSubT = allocateHostMemoryForTensor(subT, "subT", graph, uploadProg,
                                            downloadProg, tmap);

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (profile) {
    tempDir.emplace(TempDir::create());
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    engineOptions.set("autoReport.directory", tempDir->getPath());
  }
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, engineOptions);
  attachStreams(engine, tmap);

  std::vector<boost::multi_array<double, 1>> hostNZBuckets;
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    hostNZBuckets.emplace_back(
        boost::extents[blockElems * sum(subGroupNumElems[bucket])]);
  }
  boost::multi_array<unsigned, 1> hostOffsets(boost::extents[numOffsets]);
  boost::multi_array<double, 2> hostSubT(
      boost::extents[subTShape[0]][subTShape[1]]);

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    writeRandomValues(target, nzType, hostNZBuckets[bucket], -1.0, +1.0,
                      randomEngine);
    copy(target, hostNZBuckets[bucket], nzType, rawHostNZBuckets[bucket].get());
  }
  writeRandomValues(target, inputType, hostOffsets, 0u,
                    static_cast<unsigned>(baseTShape[0]) - 1, randomEngine);

  if (initialiseSubT) {
    writeRandomValues(target, inputType, hostSubT, -1.0, +1.0, randomEngine);
  }

  copy(target, hostOffsets, UNSIGNED_INT, rawHostOffsets.get());
  copy(target, hostSubT, inputType, rawHostSubT.get());

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    copy(target, hostMetaInfoBuckets[bucket], metaInfoType,
         rawHostMetaInfoBuckets[bucket].get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.setPrintStream(std::cerr);
    engine.run();
  });

  // Get the raw NZ data - used to check the updateAdd vertex
  auto ipuResultNZBuckets = hostNZBuckets;
  for (unsigned i = 0; i < numBuckets; i++) {
    copy(target, nzType, rawHostNZBuckets.at(i).get(),
         ipuResultNZBuckets.at(i));
  }
  // Get the extracted dense row data
  boost::multi_array<double, 2> ipuSubT(
      boost::extents[subTShape[0]][subTShape[1]]);
  copy(target, inputType, rawHostSubT.get(), ipuSubT);

  if (debugPrint) {
    std::cout << "ipu subT sliced results:\n";
    for (unsigned i = 0; i < subTShape[0]; i++) {
      std::cout << "\n" << i << " index into baseT:" << hostOffsets[i] << " = ";
      for (unsigned j = 0; j < subTShape[1]; j++) {
        std::cout << ipuSubT[i][j] << ",";
      }
    }
    std::cout << "\n";
  }

  // We use a dense matrix a to model this - expand out the whole sparse input,
  // keep track of which are genuine NZ values (Don't rely on them being != 0)
  auto sparseToDense =
      [&](const std::vector<boost::multi_array<double, 1>> &inputNZBuckets) {
        boost::multi_array<boost::optional<double>, 2> dense(
            boost::extents[baseTShape[0]][baseTShape[1]]);

        std::size_t nzOffset = 0;
        for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
          for (const auto &idx : processedSubGroupIndices[bucket]) {
            assert(idx < subGroupNumElems[bucket].size());
            const auto bucketNzOffset =
                blockElems *
                std::accumulate(subGroupNumElems[bucket].begin(),
                                subGroupNumElems[bucket].begin() + idx,
                                std::size_t(0));
            for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
              assert(bucketNzOffset + i <
                     inputNZBuckets[bucket].num_elements());

              if (doElementWiseTest) {
                const auto row = sparseIndices.at(nzOffset + i)[0] + rowOffset;
                if (row < baseTShape[0]) {
                  dense[row][sparseIndices.at(nzOffset + i)[1]] =
                      inputNZBuckets[bucket][bucketNzOffset + i];
                }
              } else {
                for (unsigned j = 0; j != blockSize.val[0]; ++j) {
                  for (unsigned k = 0; k != blockSize.val[1]; ++k) {
                    const auto row =
                        sparseIndices.at(nzOffset + i)[0] + j + rowOffset;
                    if (row < baseTShape[0]) {
                      auto srcOffset = bucketNzOffset + i * blockElems +
                                       j * blockSize.val[1] + k;
                      dense[row][sparseIndices.at(nzOffset + i)[1] + k] =
                          inputNZBuckets[bucket][srcOffset];
                    }
                  }
                }
              }
            }
            nzOffset += subGroupNumElems.at(bucket).at(idx);
          }
        }
        return dense;
      };
  // Model the result for slice - extract the data that is referenced
  // for update - write into the dense result.
  // In summary:
  // update : hostBaseTDense expanded from the original sparse data and
  //          is updated (where sparse data is valid) on the host
  //          ipuBaseT is read from the IPU (and should be updated)
  //          hostSubT is unchanged
  //          ipuSubT is read from the IPU (and should be unchanged)
  //
  // noupdate : hostBaseTDense expanded from the original sparse data
  //            ipuBaseT is read from the IPU (and should be unchanged)
  //            hostSubT is poplulated with the NZ values from hostBaseTDense,
  //            and left with initial values elsewhere
  //            ipuSubT is read from the IPU (and should be populated)
  //
  // In either case we should have hostSubT == ipuSubT and hostBaseT == ipuBaseT
  auto hostBaseTDense = sparseToDense(hostNZBuckets);
  auto ipuBaseTDense = sparseToDense(ipuResultNZBuckets);
  for (unsigned index = 0; index < numOffsets; index++) {
    for (unsigned i = 0; i < subTShape[1]; i++) {
      if (hostBaseTDense[hostOffsets[index]][i]) {
        if (updateAdd) {
          hostBaseTDense[hostOffsets[index]][i].get() +=
              scale * hostSubT[index][i];
        } else {
          hostSubT[index][i] = hostBaseTDense[hostOffsets[index]][i].get();
        }
      }
    }
  }

  if (debugPrint) {
    auto printDense =
        [=](boost::multi_array<boost::optional<double>, 2> &dense) {
          for (unsigned i = 0; i < baseTShape[0]; i++) {
            std::cout << "[" << i << "]:";
            for (unsigned j = 0; j < baseTShape[1]; j++) {
              if (dense[i][j]) {
                std::cout << dense[i][j].get() << ",";
              } else {
                std::cout << "x,";
              }
            }
            std::cout << "\n";
          }
        };
    if (updateAdd) {
      std::cout << "\nDense ipu updated baseT:\n";
      printDense(ipuBaseTDense);
      std::cout << "\nDense host updated baseT:\n";
      printDense(hostBaseTDense);
    } else {
      std::cout << "Dense input (baseT):\n";
      printDense(hostBaseTDense);
    }
  }

  auto extractData =
      [=](boost::multi_array<boost::optional<double>, 2> &dense) {
        boost::multi_array<double, 2> dataOnly(
            boost::extents[baseTShape[0]][baseTShape[1]]);
        for (unsigned i = 0; i < baseTShape[0]; i++) {
          for (unsigned j = 0; j < baseTShape[1]; j++) {
            dataOnly[i][j] = dense[i][j] ? dense[i][j].get() : 0.0;
          }
        }
        return dataOnly;
      };

  if (profile) {
    engine.printProfileSummary(
        std::cerr,
        {{"showExecutionSteps", (showExecutionSteps ? "true" : "false")},
         {"showVarStorage", (showVarStorage ? "true" : "false")}});
  }

  double relativeTolerance = inputType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
  double absoluteTolerance = inputType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
  if (!ignoreData) {
    bool subTMatchesModel = checkIsClose("subT", ipuSubT, hostSubT,
                                         relativeTolerance, absoluteTolerance);
    bool baseTMatchesModel = checkIsClose("baseT", extractData(ipuBaseTDense),
                                          extractData(hostBaseTDense),
                                          relativeTolerance, absoluteTolerance);
    if (!(subTMatchesModel && baseTMatchesModel)) {
      std::cerr << "Validation failed\n";
      return 1;
    }
  }
  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
