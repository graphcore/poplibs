// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplibs_support/TestDevice.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/codelets.hpp>

#include <poplibs_support/logging.hpp>

#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/Util.hpp>

#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <random>

using namespace poplar;
using namespace poplibs_test::util;
using namespace poplar::program;
using namespace poplibs_support;

// Default tolerances used in tests
constexpr double FLOAT_REL_TOL = 0.01;
constexpr double HALF_REL_TOL = 0.1;
constexpr double FLOAT_ABS_TOL = 1e-6;
constexpr double HALF_ABS_TOL = 1e-5;

enum class Pass : std::uint8_t { FWD, WU, BOTH };

std::ostream &operator<<(std::ostream &os, const Pass p) {
  switch (p) {
  case Pass::FWD:
    return os << "fwd";
  case Pass::WU:
    return os << "wu";
  case Pass::BOTH:
    return os << "both";
  }

  throw poputil::poplibs_error("Invalid pass");
}

std::istream &operator>>(std::istream &is, Pass &p) {
  std::string token;
  is >> token;

  if (token == "fwd") {
    p = Pass::FWD;
  } else if (token == "wu") {
    p = Pass::WU;
  } else if (token == "both") {
    p = Pass::BOTH;
  } else {
    throw poputil::poplibs_error("Invalid token for pass: " + token);
  }

  return is;
}

bool passEnabled(const Pass opt, const Pass pass) {
  return opt == pass || opt == Pass::BOTH;
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
    bool profile = false;
    bool showExecutionSteps = false;
    bool showVarStorage = false;
    boost::optional<std::string> profileDir;

    DeviceType deviceType = DeviceType::IpuModel2;
    unsigned numIPUs = 1;
    boost::optional<unsigned> tilesPerIPU = boost::none;

    Type dataType = HALF;
    Type indicesType = UNSIGNED_INT;
    unsigned grainSize;
    ShapeOption<std::size_t> shape;
    ShapeOption<std::size_t> numIndices;
    double scale = 1.;

    bool indicesAreSorted = false;

    bool useEmbeddingPlan = true;
    boost::optional<double> availableMemoryProportion;
    boost::optional<Type> partialType;
    std::string planConstraints;
    std::string planConstraintsFile;

    std::string sliceOptionsString;

    Pass pass = Pass::BOTH;
    bool ignoreData;
  };

  Options opts;
  po::options_description desc("embedding_layer options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("profile",
     po::value<bool>(&opts.profile)->default_value(opts.profile),
     "Output profiling report")
    ("indices-are-sorted",
     po::value<bool>(&opts.indicesAreSorted)
      ->default_value(opts.indicesAreSorted),
     "Indices are sorted (requires plan to be enabled)")
    ("profile-dir",
     po::value<boost::optional<std::string>>(&opts.profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("partial-type",
     po::value<boost::optional<Type>>(&opts.partialType)
      ->default_value(boost::none),
     "Partials type (defaults to data type ")     
    ("show-execution-steps",
     po::value<bool>(&opts.showExecutionSteps)
       ->default_value(opts.showExecutionSteps),
     "Show execution steps (requires profiling)")
    ("show-var-storage",
     po::value<bool>(&opts.showVarStorage)->default_value(opts.showVarStorage),
     "Show variable liveness (requires profiling)")
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
     "The data type of values stored in the embedding matrix")
    ("grain-size",
     po::value<unsigned>(&opts.grainSize),
     "Minimum elements per slice mapped to each tile. Defaults to the vector "
     "width of the data type for the target chosen.")
    ("shape",
     po::value<ShapeOption<std::size_t>>(&opts.shape)->required(),
     "The shape of the embedding matrix, must be 2D.")
    ("num-indices",
     po::value<ShapeOption<std::size_t>>(&opts.numIndices)->required(),
     "The amount of indices to use. Can be a list which specifies multiple "
     "numbers of indices which will be used to index/update the embedding "
     "matrix")
    ("indices-type",
     po::value<Type>(&opts.indicesType)->default_value(opts.indicesType),
     "The data type of the indices.")
    ("scale",
     po::value<double>(&opts.scale)->default_value(opts.scale),
     "Scale applied to the deltas during the update pass")
    ("pass",
     po::value<Pass>(&opts.pass)->default_value(opts.pass),
     "Which pass of the embedding layer to perform: fwd | wu | both")
    ("ignore-data",
     "Don't upload and download the results from the device. Note that this "
     "means the result is not validated against the model.")
    ("use-embedding-plan",
     po::value<bool>(&opts.useEmbeddingPlan)->default_value(
       opts.useEmbeddingPlan
     ),
     "Create and use plan for embedding layer rather than default slice "
     "implementation.")
    ("available-memory-proportion",
     po::value<boost::optional<double>>(&opts.availableMemoryProportion),
     "Proportion of memory available for temporary memory usage in the "
     "operation when a plan is used")
    ("plan-constraints",
     po::value<std::string>(&opts.planConstraints),
     "Constraints on the plan for the embedding as a JSON string")
    ("plan-constraints-file",
     po::value<std::string>(&opts.planConstraintsFile),
     "Constraints on the plan for the embedding as a path to a JSON file")
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

  if (opts.shape->size() != 2) {
    throw poputil::poplibs_error("The embedding matrix must be 2 dimensions");
  }

  loadConstraintsFromFile(opts.planConstraintsFile, opts.planConstraints);

  const bool compileIPUCode = true;
  auto device = createTestDevice(opts.deviceType, opts.numIPUs,
                                 opts.tilesPerIPU, compileIPUCode);

  const auto &target = device.getTarget();

  if (!vm.count("grain-size")) {
    opts.grainSize = target.getVectorWidth(opts.dataType);
  }

  if (vm.count("grain-size") && opts.useEmbeddingPlan) {
    throw std::logic_error("Both grain-size and use-embedding-plan specified "
                           "but are mutually exclusive");
  }

  opts.ignoreData = vm.count("ignore-data");

  const std::vector<std::size_t> &numIndices = opts.numIndices;

  logging::popops::info("Embedding matrix shape: {}", opts.shape);
  logging::popops::info("Number of indices to process: {}", numIndices);
  logging::popops::info("Performing pass: {}", opts.pass);

  Graph graph(target);
  popops::addCodelets(graph);

  Sequence prog, uploadProg, downloadProg;

  std::mt19937 randomEngine;
  std::vector<std::pair<std::string, char *>> tmap;

  OptionFlags sliceOptions;
  if (opts.availableMemoryProportion) {
    sliceOptions.set("availableMemoryProportion",
                     std::to_string(*opts.availableMemoryProportion));
  }
  if (!opts.planConstraints.empty()) {
    sliceOptions.set("planConstraints", opts.planConstraints);
  }
  sliceOptions.set("usedForSlice",
                   passEnabled(opts.pass, Pass::FWD) ? "true" : "false");
  sliceOptions.set("usedForUpdate",
                   passEnabled(opts.pass, Pass::WU) ? "true" : "false");
  sliceOptions.set("indicesAreSorted",
                   opts.indicesAreSorted && opts.useEmbeddingPlan ? "true"
                                                                  : "false");
  if (opts.partialType) {
    sliceOptions.set("partialType", opts.partialType->toString());
  }

  if (!opts.sliceOptionsString.empty()) {
    std::stringstream ss(opts.sliceOptionsString);
    poplar::readJSON(ss, sliceOptions);
  }

  popops::SlicePlan plan;
  Tensor embeddingMatrix;
  if (opts.useEmbeddingPlan) {
    logging::popops::info("Graph construction: Planning embedding layer");
    plan = popops::embedding::plan(graph, opts.dataType, opts.shape[0],
                                   opts.shape[1], numIndices, sliceOptions);
    logging::popops::info("Graph construction: create embedding matrix");
    embeddingMatrix =
        popops::createSliceableTensor(graph, opts.dataType, opts.shape, {0},
                                      {1}, plan, sliceOptions, "embedding");
  } else {
    logging::popops::info(
        "Graph construction: create embedding matrix, grain size {}",
        opts.grainSize);
    embeddingMatrix =
        popops::createSliceableTensor(graph, opts.dataType, opts.shape, {0},
                                      {1}, opts.grainSize, "embedding");
  }

  const auto rawEmbeddingMatrix =
      allocateHostMemoryForTensor(embeddingMatrix, "embeddingMatrix", graph,
                                  uploadProg, downloadProg, tmap);

  struct PerIndexSet {
    // The indices
    std::vector<unsigned> hostIdxs;
    Tensor idxs;
    std::unique_ptr<char[]> rawIdxs;

    // The output from slicing using these indices
    boost::multi_array<double, 2> hostOut;
    std::unique_ptr<char[]> rawOut;
    // The data updated to these indices of the matrix
    boost::multi_array<double, 2> hostDeltas;
    std::unique_ptr<char[]> rawDeltas;
  };

  std::vector<PerIndexSet> perIndexSet(numIndices.size());
  for (std::size_t i = 0; i < numIndices.size(); ++i) {
    perIndexSet[i].hostIdxs.resize(numIndices[i]);
    const auto activationExtents =
        boost::extents[numIndices[i]][opts.shape->at(1)];
    perIndexSet[i].hostOut.resize(activationExtents);
    perIndexSet[i].hostDeltas.resize(activationExtents);

    writeRandomValues(target, opts.indicesType, perIndexSet[i].hostIdxs, 0u,
                      static_cast<unsigned>(opts.shape->at(0) - 1),
                      randomEngine);
    if (opts.indicesAreSorted) {
      std::sort(perIndexSet[i].hostIdxs.begin(), perIndexSet[i].hostIdxs.end());
    }

    const std::string handle = "indices_" + std::to_string(i);
    logging::popops::trace("Indices[{}]: {}", i, perIndexSet[i].hostIdxs);
    perIndexSet[i].idxs = createIndicesTensor(graph, {0}, numIndices[i], plan,
                                              sliceOptions, handle);
    perIndexSet[i].rawIdxs = allocateHostMemoryForTensor(
        perIndexSet[i].idxs, handle, graph, uploadProg, downloadProg, tmap);
  }

  if (passEnabled(opts.pass, Pass::FWD)) {
    for (std::size_t i = 0; i < numIndices.size(); ++i) {
      logging::popops::info("Graph construction: create gather operation {}",
                            i);
      const std::string handle = "extracted_" + std::to_string(i);
      const auto extractedData =
          popops::multiSlice(graph, embeddingMatrix, perIndexSet[i].idxs, {0},
                             {1}, prog, plan, sliceOptions, handle);
      perIndexSet[i].rawOut = allocateHostMemoryForTensor(
          extractedData, handle, graph, uploadProg, downloadProg, tmap);
    }
  }

  if (passEnabled(opts.pass, Pass::WU)) {
    const auto scale =
        graph.addConstant(opts.dataType, {}, opts.scale, "scale");
    graph.setTileMapping(scale, 0);
    for (std::size_t i = 0; i < numIndices.size(); ++i) {
      logging::popops::info("Graph construction: create update operation");
      Tensor deltas;
      const std::string handle = "deltas_" + std::to_string(i);
      if (opts.useEmbeddingPlan) {
        deltas = popops::createSliceTensor(graph, opts.dataType, opts.shape,
                                           {0}, {1}, numIndices[i], plan,
                                           sliceOptions, handle);
      } else {
        deltas = popops::createSliceTensor(graph, embeddingMatrix, {0}, {1},
                                           numIndices[i], handle);
      }
      perIndexSet[i].rawDeltas = allocateHostMemoryForTensor(
          deltas, handle, graph, uploadProg, downloadProg, tmap);

      popops::multiUpdateAdd(graph, embeddingMatrix, deltas,
                             perIndexSet[i].idxs, scale, {0}, {1}, prog, plan,
                             sliceOptions, "updated_" + std::to_string(i));
    }
  }

  Sequence ctrlProg;
  if (!opts.ignoreData) {
    ctrlProg.add(uploadProg);
  }
  ctrlProg.add(prog);
  if (!opts.ignoreData) {
    ctrlProg.add(downloadProg);
  }

  OptionFlags engineOptions;
  if (opts.profile || opts.profileDir) {
    engineOptions.set("debug.instrumentCompute", "true");
    engineOptions.set("debug.computeInstrumentationLevel", "device");
    if (opts.profileDir) {
      engineOptions.set("autoReport.all", "true");
      engineOptions.set("autoReport.directory", *opts.profileDir);
    }
  }

  logging::popops::info("Create engine");
  Engine engine(graph, ctrlProg, engineOptions);

  if (vm.count("compile-only"))
    return 0;

  const auto embeddingMatrixExtents =
      boost::extents[opts.shape->at(0)][opts.shape->at(1)];
  boost::multi_array<double, 2> hostEmbeddingMatrix(embeddingMatrixExtents);

  if (!opts.ignoreData) {
    logging::popops::info("Generating the embedding matrix on the host");
    attachStreams(engine, tmap);

    writeRandomValues(target, opts.dataType, hostEmbeddingMatrix, -10., 10.,
                      randomEngine);
    copy(target, hostEmbeddingMatrix, opts.dataType, rawEmbeddingMatrix.get());
    for (std::size_t i = 0; i < numIndices.size(); ++i) {
      copy(target, perIndexSet[i].hostIdxs.data(),
           perIndexSet[i].hostIdxs.size(), UNSIGNED_INT,
           perIndexSet[i].rawIdxs.get());
      if (passEnabled(opts.pass, Pass::WU)) {
        writeRandomValues(target, opts.dataType, perIndexSet[i].hostDeltas, -1.,
                          1., randomEngine);
        copy(target, perIndexSet[i].hostDeltas, opts.dataType,
             perIndexSet[i].rawDeltas.get());
      }
    }
  }

  logging::popops::info("Run program");
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  bool matchesModel = true;

  if (!opts.ignoreData) {
    const double absTol = opts.dataType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    const double relTol = opts.dataType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;

    boost::multi_array<double, 2> modelEmbeddingMatrix(embeddingMatrixExtents);
    std::copy_n(hostEmbeddingMatrix.data(), hostEmbeddingMatrix.num_elements(),
                modelEmbeddingMatrix.data());
    for (std::size_t i = 0; i < numIndices.size(); ++i) {
      boost::multi_array<double, 2> modelExtractedData(
          boost::extents[numIndices[i]][opts.shape->at(1)]);

      if (passEnabled(opts.pass, Pass::FWD)) {
        logging::popops::info("Validate gather operation against model");
        poplibs_test::embedding::multiSlice(
            hostEmbeddingMatrix, perIndexSet[i].hostIdxs, modelExtractedData);

        copy(target, opts.dataType, perIndexSet[i].rawOut.get(),
             perIndexSet[i].hostOut);
        matchesModel &= checkIsClose("multiSlice_" + std::to_string(i),
                                     perIndexSet[i].hostOut, modelExtractedData,
                                     relTol, absTol);
      }

      if (passEnabled(opts.pass, Pass::WU)) {
        logging::popops::info("Validate update operation against model");

        poplibs_test::embedding::multiUpdateAdd(
            perIndexSet[i].hostDeltas, perIndexSet[i].hostIdxs, opts.scale,
            modelEmbeddingMatrix);
      }
    }
    copy(target, opts.dataType, rawEmbeddingMatrix.get(), hostEmbeddingMatrix);
    matchesModel &= checkIsClose("multiUpdateAdd", hostEmbeddingMatrix,
                                 modelEmbeddingMatrix, relTol, absTol);
  }

  if (opts.profile) {
    engine.printProfileSummary(
        std::cout,
        {
            {"showExecutionSteps", opts.showExecutionSteps ? "true" : "false"},
            {"showVarStorage", opts.showVarStorage ? "true" : "false"},
        });
  }

  if (!matchesModel) {
    std::cerr << "Validation failed" << std::endl;
  }

  return matchesModel ? 0 : 1;
}
