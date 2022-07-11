// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <iostream>
#include <numeric>
#include <poplar/Engine.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <poplar_test/Util.hpp>
#include <poplibs_support/MultiArray.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/logging.hpp>
#include <poplibs_support/print.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Operation.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <optional>
#include <vector>

using namespace poplar;
using namespace poplar::program;
using namespace poplar_test;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;
using namespace poplibs_test::util;
using poplibs_support::toString;

MultiArray<unsigned int> createHostIndices(const Target &target,
                                           std::mt19937 &randomEngine,
                                           unsigned numIndicesPerGroup,
                                           unsigned sliceDimSize,
                                           unsigned groupSize) {
  MultiArray<unsigned int> indices{groupSize, numIndicesPerGroup};
  writeRandomValues(target, UNSIGNED_INT, indices, 0U, sliceDimSize - 1,
                    randomEngine);
  return indices;
}

static bool
multiSlice(const DeviceType &deviceType, const unsigned numIPUs,
           const boost::optional<unsigned> &tilesPerIPU,
           const std::vector<std::size_t> baseShape, const unsigned numIndices,
           const boost::optional<unsigned> grainSize, const Type &dataType,
           boost::optional<unsigned> groupSize_, const bool compileOnly,
           OptionFlags &sliceOptions, OptionFlags &engineOptions) {
  const unsigned D = baseShape.at(0);
  const unsigned E = baseShape.at(1);
  const auto grouped = groupSize_ != boost::none;
  const unsigned groupSize = groupSize_ == boost::none ? 1 : *groupSize_;

  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};

  if (grainSize != boost::none) {
    sliceOptions.set("grainSize", std::to_string(*grainSize));
  }

  const bool compileIPUCode = true;
  auto device =
      createTestDevice(deviceType, numIPUs, tilesPerIPU, compileIPUCode);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  logging::popops::info("Running on {} with {} ipus and {} tiles on each IPU",
                        deviceType, numIPUs, target.getTilesPerIPU());
  logging::popops::info("Base matrix shape: {} with group size {}, "
                        "num Indices {}",
                        baseShape, groupSize, numIndices);
  auto plan = grouped ? embedding::plan(graph, dataType, groupSize, D, E,
                                        {numIndices}, sliceOptions)
                      : embedding::plan(graph, dataType, D, E, {numIndices},
                                        sliceOptions);

  // Map the tensor carefully to ensure balance and minimise edge pointers
  Tensor t = grouped
                 ? createGroupedSliceableTensor(graph, dataType, groupSize,
                                                {D, E}, sliceDims, sliceSizes,
                                                plan, sliceOptions, "t")
                 : createSliceableTensor(graph, dataType, {D, E}, sliceDims,
                                         sliceSizes, plan, sliceOptions, "t");

  Sequence prog;

  std::mt19937 randomEngine;
  auto indices =
      createHostIndices(target, randomEngine, numIndices, D, groupSize);

  Tensor offset =
      grouped
          ? createGroupedIndicesTensor(graph, groupSize, sliceDims, numIndices,
                                       plan, sliceOptions, "offset")
          : createIndicesTensor(graph, sliceDims, numIndices, plan,
                                sliceOptions, "offset");

  auto s = grouped
               ? groupedMultiSlice(graph, t, offset, sliceDims, sliceSizes,
                                   prog, plan, sliceOptions, "MultisliceTest")
               : multiSlice(graph, t, offset, sliceDims, sliceSizes, prog, plan,
                            sliceOptions, "MultisliceTest");

  const auto indicesType = offset.elementType();
  graph.createHostWrite("inOffset", offset, true);
  graph.createHostWrite("in", t, true);
  graph.createHostRead("out", s, true);
  MultiArray<double> hIn{groupSize, D, E};
  MultiArray<double> hOut{groupSize, numIndices, E};

  // random integers
  writeRandomValues(target, dataType, hIn, 0., 1., randomEngine);
  std::transform(hIn.data(), hIn.data() + hIn.numElements(), hIn.data(),
                 [](double x) { return static_cast<int>(x * 16) % 16; });

  std::vector<char> rawOffsets(target.getTypeSize(indicesType) *
                               indices.numElements());
  std::vector<char> rawHIn(target.getTypeSize(dataType) * hIn.numElements());
  std::vector<char> rawHOut(target.getTypeSize(dataType) * hOut.numElements());

  copy(target, indices, indicesType, rawOffsets.data());

  const auto metadata = QuarterMetadata(QuarterMetadata::Format::F152, 1);
  if (dataType.requiresMetadata()) {
    copy(target, hIn.data(), hIn.numElements(), dataType, metadata,
         rawHIn.data());
  } else {
    copy(target, hIn, dataType, rawHIn.data());
  }
  // Engine creation will fail for non-cpu targets if many edge pointers or
  // significant exchange is required; this should not happen if
  // createSliceableTensor() has given a good layout
  Engine eng(graph, prog, engineOptions);

  if (compileOnly) {
    return true;
  }

  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("inOffset", rawOffsets.data(),
                    rawOffsets.data() + rawOffsets.size());
    eng.writeTensor("in", rawHIn.data(), rawHIn.data() + rawHIn.size());
    eng.run();
    eng.readTensor("out", rawHOut.data(), rawHOut.data() + rawHOut.size());
  });
  QuarterMetadata metadataOut;
  if (dataType.requiresMetadata()) {
    copy(target, dataType, metadataOut, rawHOut.data(), hOut.data(),
         hOut.numElements());
  } else {
    copy(target, dataType, rawHOut.data(), hOut);
  }
  bool matches = true;
  for (unsigned g = 0; g != groupSize; ++g) {
    for (unsigned i = 0; i != numIndices; ++i) {
      auto d = indices[g][i];
      for (unsigned elem = 0; elem != E; ++elem) {
        auto expected = hIn[g][d][elem];
        auto actual = hOut[g][i][elem];
        if (expected != actual) {
          matches = false;
          std::cerr << "Mismatch at [" << g << "][" << i << "][" << elem;
          std::cerr << "] at index " << d << " : " << expected << "(exp) ";
          std::cerr << actual << "(actual)\n";
        }
      }
    }
  }
  return matches;
}

static bool
multiUpdate(const DeviceType &deviceType, const unsigned numIPUs,
            const boost::optional<unsigned> &tilesPerIPU,
            const std::vector<std::size_t> baseShape, const unsigned numIndices,
            const boost::optional<unsigned> grainSize, const Type &dataType,
            boost::optional<unsigned> groupSize_, boost::optional<Operation> op,
            const float updateScaling, const bool compileOnly,
            OptionFlags &sliceOptions, OptionFlags &engineOptions) {
  const bool updateOp = op != boost::none;
  const bool opUsesScale = updateOp && *op == popops::Operation::ADD;
  const bool useFloatScalingForHalf = false;
  const Type scaleTensorType =
      dataType == HALF && useFloatScalingForHalf ? FLOAT : dataType;

  const unsigned D = baseShape.at(0);
  const unsigned E = baseShape.at(1);
  const auto grouped = groupSize_ != boost::none;
  const unsigned groupSize = groupSize_ == boost::none ? 1 : *groupSize_;

  std::vector<std::size_t> sliceDims{0};
  std::vector<std::size_t> sliceSizes{1};

  const bool compileIPUCode = true;
  auto device =
      createTestDevice(deviceType, numIPUs, tilesPerIPU, compileIPUCode);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  Tensor scale;
  // call both overloads to check the API
  auto plan = grouped ? embedding::plan(graph, dataType, groupSize, D, E,
                                        {numIndices}, sliceOptions)
                      : embedding::plan(graph, dataType, D, E, {numIndices},
                                        sliceOptions);

  auto t = grouped ? createGroupedSliceableTensor(graph, dataType, groupSize,
                                                  {D, E}, sliceDims, sliceSizes,
                                                  plan, sliceOptions, "t")
                   : createSliceableTensor(graph, dataType, {D, E}, sliceDims,
                                           sliceSizes, plan, sliceOptions, "t");

  auto s =
      grouped
          ? createGroupedSliceTensor(graph, dataType, groupSize, {D, E},
                                     sliceDims, sliceSizes, numIndices, plan,
                                     sliceOptions, "s")
          : createSliceTensor(graph, dataType, {D, E}, sliceDims, sliceSizes,
                              numIndices, plan, sliceOptions, "s");

  std::mt19937 randomEngine;
  auto indices =
      createHostIndices(target, randomEngine, numIndices, D, groupSize);
  Sequence prog;
  if (dataType.requiresMetadata()) {
    auto metadata =
        createConstantMetadataTensor(graph, QuarterMetadata::Format::F152, 1);
    prog.add(Copy(metadata, t.getMetadata()));
  }
  Tensor offset =
      grouped
          ? createGroupedIndicesTensor(graph, groupSize, sliceDims, numIndices,
                                       plan, sliceOptions, "offset")
          : createIndicesTensor(graph, sliceDims, numIndices, plan,
                                sliceOptions, "offset");

  if (!updateOp) {
    grouped ? groupedMultiUpdate(graph, t, s, offset, sliceDims, sliceSizes,
                                 prog, plan, sliceOptions, "MultiUpdateTest")
            : multiUpdate(graph, t, s, offset, sliceDims, sliceSizes, prog,
                          plan, sliceOptions, "MultiUpdateTest");
  } else {
    if (*op == popops::Operation::ADD) {
      scale = graph.addVariable(scaleTensorType, {}, "scale");
      graph.setTileMapping(scale, 0);
      grouped
          ? groupedMultiUpdateAdd(graph, t, s, offset, scale, sliceDims,
                                  sliceSizes, prog, plan, sliceOptions,
                                  "MultiUpdateTest")
          : multiUpdateAdd(graph, t, s, offset, scale, sliceDims, sliceSizes,
                           prog, plan, sliceOptions, "MultiUpdateTest");
    } else if (*op == popops::Operation::MAX) {
      grouped
          ? groupedMultiUpdateMax(graph, t, s, offset, sliceDims, sliceSizes,
                                  prog, plan, sliceOptions, "MultiUpdateTest")
          : multiUpdateMax(graph, t, s, offset, sliceDims, sliceSizes, prog,
                           plan, sliceOptions, "MultUpdateTest");
    } else {
      std::cerr << "\n Unsupported op in multiUpdateOp\n";
      return 0;
    }
  }

  const auto indicesType = offset.elementType();
  graph.createHostWrite("inOffset", offset, true);
  graph.createHostWrite("inS", s, true);
  graph.createHostWrite("inT", t, true);
  graph.createHostRead("outT", t, true);
  if (updateOp && opUsesScale)
    graph.createHostWrite("scale", scale, true);

  const MultiArrayShape hOutShape = {groupSize, D, E};
  MultiArray<double> hIn{groupSize, numIndices, E};

  MultiArray<double> hOut{hOutShape};
  MultiArray<double> expected{hOutShape};

  // random integers
  writeRandomValues(target, dataType, hIn, -1., 1., randomEngine);
  std::transform(hIn.data(), hIn.data() + hIn.numElements(), hIn.data(),
                 [](double x) { return static_cast<int>(x * 16) % 16; });
  writeRandomValues(target, dataType, hOut, -1., 1., randomEngine);
  std::transform(hOut.data(), hOut.data() + hOut.numElements(), hOut.data(),
                 [](double x) { return static_cast<int>(x * 16) % 16; });

  // copy base to expected
  std::copy(hOut.data(), hOut.data() + hOut.numElements(), expected.data());

  std::vector<char> rawOffsets(target.getTypeSize(indicesType) *
                               indices.numElements());
  std::vector<char> rawIn(target.getTypeSize(dataType) * hIn.numElements());
  std::vector<char> rawOut(target.getTypeSize(dataType) * hOut.numElements());
  std::vector<char> rawScaleIn(target.getTypeSize(scaleTensorType));
  const auto metadata = QuarterMetadata(QuarterMetadata::Format::F143, 2);
  if (scaleTensorType.requiresMetadata()) {
    MultiArray<float> scalingF{1};
    scalingF[0] = updateScaling;
    copy(target, scalingF.data(), scalingF.numElements(), scaleTensorType,
         metadata, rawScaleIn.data());
  } else {
    MultiArray<float> scalingF{1};
    scalingF[0] = updateScaling;
    copy(target, scalingF, scaleTensorType, rawScaleIn.data());
  }
  copy(target, indices, indicesType, rawOffsets.data());
  if (dataType.requiresMetadata()) {
    copy(target, hIn.data(), hIn.numElements(), dataType, metadata,
         rawIn.data());
    copy(target, hOut.data(), hOut.numElements(), dataType, metadata,
         rawOut.data());
  } else {
    copy(target, hIn, dataType, rawIn.data());
    copy(target, hOut, dataType, rawOut.data());
  }
  Engine eng(graph, prog, engineOptions);
  device.bind([&](const Device &d) {
    eng.load(d);
    if (updateOp && opUsesScale) {
      eng.writeTensor("scale", rawScaleIn.data(),
                      rawScaleIn.data() + rawScaleIn.size());
    }
    eng.writeTensor("inOffset", rawOffsets.data(),
                    rawOffsets.data() + rawOffsets.size());
    eng.writeTensor("inT", rawOut.data(), rawOut.data() + rawOut.size());
    eng.writeTensor("inS", rawIn.data(), rawIn.data() + rawIn.size());
    eng.run();
    eng.readTensor("outT", rawOut.data(), rawOut.data() + rawOut.size());
  });
  QuarterMetadata metadataOut;
  if (dataType.requiresMetadata()) {
    copy(target, dataType, metadataOut, rawOut.data(), hOut.data(),
         hOut.numElements());
  } else {
    copy(target, dataType, rawOut.data(), hOut);
  }

  for (unsigned g = 0; g != groupSize; ++g) {
    for (unsigned i = 0; i != numIndices; ++i) {
      auto d = indices[g][i];
      for (unsigned elem = 0; elem != E; ++elem) {
        if (!updateOp) {
          expected[g][d][elem] = hIn[g][i][elem];
        } else {
          if (*op == popops::Operation::ADD) {
            expected[g][d][elem] += updateScaling * hIn[g][i][elem];
          } else if (*op == popops::Operation::MAX) {
            expected[g][d][elem] =
                std::max(expected[g][d][elem], hIn[g][i][elem]);
          }
        }
      }
    }
  }

  // validate
  bool matches = true;
  for (unsigned g = 0; g != groupSize; ++g) {
    for (unsigned d = 0; d != D; ++d) {
      for (unsigned elem = 0; elem != E; ++elem) {
        if (expected[g][d][elem] != hOut[g][d][elem]) {
          matches = false;
          std::cerr << "Mismatch at [" << g << "][" << d;
          std::cerr << "][" << elem << "] : " << expected[g][d][elem];
          std::cerr << "(exp) " << hOut[g][d][elem] << "(actual)\n";
        }
      }
    }
  }
  if (dataType.requiresMetadata() && metadata != metadataOut) {
    matches = false;
  }
  return matches;
}

void loadConstraintsFromFile(const std::string &path,
                             std::string &constraints) {
  if (!path.empty()) {
    std::ifstream is(path, std::ios_base::in);
    if (!is.good()) {
      throw poputil::poplibs_error("Constraints file " + path +
                                   " could not be opened");
    }
    is.seekg(0, std::ios::end);
    const auto bytes = is.tellg();

    std::string constraints(bytes, '\0');
    is.seekg(0);
    is.read(&constraints[0], bytes);
  }
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  struct Options {
    boost::optional<std::string> profileDir;

    DeviceType deviceType = DeviceType::IpuModel2;
    unsigned numIPUs = 1;
    boost::optional<unsigned> tilesPerIPU = boost::none;
    boost::optional<unsigned> groupSize = boost::none;

    // only applicable for update
    boost::optional<Operation> operation = boost::none;
    bool update = false;

    Type dataType = HALF;
    boost::optional<unsigned> grainSize = boost::none;
    ShapeOption<std::size_t> shape;
    unsigned numIndices;
    double scale = 1.;

    bool indicesAreSorted = false;
    boost::optional<double> availableMemoryProportion;
    boost::optional<Type> partialType;
    std::string planConstraints;
    std::string planConstraintsFile;
    std::string sliceOptionsString;
  };

  Options opts;
  po::options_description desc("MultiSlice/MultiUpdate options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("indices-are-sorted",
     po::value<bool>(&opts.indicesAreSorted)
      ->default_value(opts.indicesAreSorted),
     "Indices are sorted (requires plan to be enabled)")
    ("update",
     po::value<bool>(&opts.update)->default_value(opts.update),
     "Do a multi-update operation, otherwise a multi-slice operation")
    ("profile-dir",
     po::value<boost::optional<std::string>>(&opts.profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("group-size",
     po::value<boost::optional<unsigned>>(&opts.groupSize)
      ->default_value(boost::none),
     "Group size to use (if not specified, no grouping is used).")
    ("partial-type",
     po::value<boost::optional<Type>>(&opts.partialType)
      ->default_value(boost::none),
     "Partials type (defaults to data type ")
    ("device-type",
     po::value<DeviceType>(&opts.deviceType)->default_value(opts.deviceType),
     deviceTypeHelp)
    ("ipus",
     po::value<unsigned>(&opts.numIPUs)->default_value(opts.numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu",
     po::value<boost::optional<unsigned>>(&opts.tilesPerIPU),
     "Number of tiles per IPU")
    ("data-type",
     po::value<Type>(&opts.dataType)->default_value(opts.dataType),
     "The data type of values stored in the base matrix")
    ("grain-size",
     po::value<boost::optional<unsigned>>(&opts.grainSize)->
       default_value(boost::none),
     "Minimum elements per slice mapped to each tile. Defaults to the vector "
     "width of the data type for the target chosen.")
    ("shape",
     po::value<ShapeOption<std::size_t>>(&opts.shape)->required(),
     "The shape of the base matrix, must be 2D.")
    ("num-indices",
     po::value<unsigned>(&opts.numIndices)->required(),
     "The amount of indices er group to use. numbers of indices which will be "
     "used to index/update the base matrix")
    ("operation", po::value<boost::optional<Operation>>(&opts.operation)
                  ->default_value(boost::none),
      "The operation to perform (ADD, MAX) when update = true Defaults to a "
      "plain update")
    ("scale",
     po::value<double>(&opts.scale)->default_value(opts.scale),
     "Scale applied for update when the operation is ADD")
    ("available-memory-proportion",
     po::value<boost::optional<double>>(&opts.availableMemoryProportion),
     "Proportion of memory available for temporary memory usage in the "
     "operation when a plan is used")
    ("plan-constraints",
     po::value<std::string>(&opts.planConstraints),
     "Constraints on the plan for the multislice/update as a JSON string")
    ("plan-constraints-file",
     po::value<std::string>(&opts.planConstraintsFile),
     "Constraints on the plan for the multislice/update as a path to a JSON "
     "file")
    ("slice-options",
     po::value<std::string>(&opts.sliceOptionsString),
     "String with JSON formatted options to pass to the slice operation")
    ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  const bool compileOnly = (vm.count("compile-only"));

  if (opts.shape->size() != 2) {
    throw poputil::poplibs_error("The base matrix must be 2 dimensions");
  }

  loadConstraintsFromFile(opts.planConstraintsFile, opts.planConstraints);

  const std::vector<std::size_t> &baseShape = opts.shape;

  OptionFlags sliceOptions;
  if (opts.availableMemoryProportion) {
    sliceOptions.set("availableMemoryProportion",
                     std::to_string(*opts.availableMemoryProportion));
  }
  if (!opts.planConstraints.empty()) {
    sliceOptions.set("planConstraints", opts.planConstraints);
  }
  sliceOptions.set("usedForSlice", opts.update ? "false" : "true");
  sliceOptions.set("usedForUpdate", opts.update ? "true" : "false");
  sliceOptions.set("indicesAreSorted",
                   opts.indicesAreSorted ? "true" : "false");
  if (opts.partialType) {
    sliceOptions.set("partialType", opts.partialType->toString());
  }

  if (opts.update) {
    std::string opName;
    if (opts.operation == boost::none) {
      opName = "none";
    } else if (*opts.operation == Operation::MAX) {
      opName = "max";
    } else if (*opts.operation == Operation::ADD) {
      opName = "add";
    } else {
      std::cerr << "Invalid operation type for update"
                << "\n";
      return 0;
    }
    sliceOptions.set("operationForUpdate", opName);
  }

  if (!opts.sliceOptionsString.empty()) {
    std::stringstream ss(opts.sliceOptionsString);
    poplar::readJSON(ss, sliceOptions);
  }

  std::optional<TempDir> tempDir;
  OptionFlags engineOptions;
  if (opts.profileDir) {
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    if (opts.profileDir) {
      engineOptions.set("autoReport.directory", *opts.profileDir);
    } else {
      tempDir.emplace(TempDir::create());
      engineOptions.set("autoReport.directory", tempDir->getPath());
    }
  }

  bool passed = true;
  if (!opts.update) {
    passed =
        multiSlice(opts.deviceType, opts.numIPUs, opts.tilesPerIPU, baseShape,
                   opts.numIndices, opts.grainSize, opts.dataType,
                   opts.groupSize, compileOnly, sliceOptions, engineOptions);
  } else {
    passed = multiUpdate(opts.deviceType, opts.numIPUs, opts.tilesPerIPU,
                         baseShape, opts.numIndices, opts.grainSize,
                         opts.dataType, opts.groupSize, opts.operation,
                         opts.scale, compileOnly, sliceOptions, engineOptions);
  }
  if (!passed) {
    std::cerr << "Test failed\n";
  }
  return !passed;
}
