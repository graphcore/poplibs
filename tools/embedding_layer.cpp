#include "TestDevice.hpp"

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>

#include <popops/codelets.hpp>
#include <popops/DynamicSlice.hpp>

#include <poplibs_support/logging.hpp>

#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/Util.hpp>

#include <poputil/exceptions.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <random>

using namespace poplar;
using namespace poplibs_test::util;
using namespace poplar::program;

namespace logging = poplibs_support::logging;

// Default tolerances used in tests
constexpr double FLOAT_REL_TOL = 0.1;
constexpr double HALF_REL_TOL = 0.3;
constexpr double FLOAT_ABS_TOL = 1e-5;
constexpr double HALF_ABS_TOL = 7e-2;

enum class Pass : std::uint8_t {
  FWD, WU, BOTH
};

std::ostream &operator<<(std::ostream &os, const Pass p) {
  switch(p) {
    case Pass::FWD: return os << "fwd";
    case Pass::WU: return os << "wu";
    case Pass::BOTH: return os << "both";
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

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  struct Options {
    bool profile = false;
    bool showExecutionSteps = false;
    bool showVarStorage = false;

    DeviceType deviceType = DeviceType::IpuModel;
    unsigned numIPUs = IPUModel{}.numIPUs;
    unsigned tilesPerIPU = IPUModel{}.tilesPerIPU;

    Type dataType = FLOAT;
    Type indicesType = UNSIGNED_INT;
    unsigned grainSize;
    ShapeOption<std::size_t> shape;
    unsigned numIndices;
    double scale = 1.;
    bool useEmbeddingPlan = true;

    Pass pass = Pass::BOTH;
    bool ignoreData = false;
  };

  Options opts;

  po::options_description desc("embedding_layer options");
  desc.add_options()
    ("help", "Produce help message")
    ("profile",
     po::value<bool>(&opts.profile)->default_value(opts.profile),
     "Output profiling report")
    ("show-execution-steps",
     po::value<bool>(&opts.showExecutionSteps)
       ->default_value(opts.showExecutionSteps),
     "Show execution steps (requires profiling)")
    ("show-var-storage",
     po::value<bool>(&opts.showVarStorage)->default_value(opts.showVarStorage),
     "Show variable liveness (requires profiling)")
    ("device-type",
     po::value<DeviceType>(&opts.deviceType)->default_value(opts.deviceType),
     "Device type: Cpu | Sim | Hw | IpuModel")
    ("ipus",
     po::value<unsigned>(&opts.numIPUs)->default_value(opts.numIPUs),
     "Number of IPUs")
    ("tiles-per-ipu",
     po::value<unsigned>(&opts.tilesPerIPU)->default_value(opts.tilesPerIPU),
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
     po::value<unsigned>(&opts.numIndices)->required(),
     "The amount of indices to use")
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
     po::value<bool>(&opts.ignoreData)->default_value(opts.ignoreData),
     "Don't upload and download the results from the device. Note that this "
     "means the result is not validated against the model.")
    ("use-embedding-plan",
     po::value<bool>(&opts.useEmbeddingPlan)->default_value(
       opts.useEmbeddingPlan
     ),
     "Create and use plan for embedding layer rather than default slice "
     "implementation.")
    ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (opts.shape->size() != 2) {
    throw poputil::poplibs_error("The embedding matrix must be 2 dimensions");
  }

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

  logging::info("Embedding matrix shape: {}", opts.shape);
  logging::info("Number of indices to process: {}", opts.numIndices);
  logging::info("Performing pass: {}", opts.pass);

  Graph graph(target);
  popops::addCodelets(graph);

  Sequence prog, uploadProg, downloadProg;

  std::mt19937 randomEngine;
  std::unique_ptr<char []> rawExtractedData, rawDeltas;
  std::vector<std::pair<std::string, char *>> tmap;

  const OptionFlags sliceOptions;
  popops::SlicePlan plan;
  Tensor embeddingMatrix;
  if (opts.useEmbeddingPlan) {
    logging::info("Graph construction: Planning embedding layer");
    plan = popops::embedding::plan(graph, opts.dataType,
                                   opts.shape[0], opts.shape[1],
                                   {opts.numIndices}, sliceOptions);
    logging::info("Graph construction: create embedding matrix");
    embeddingMatrix =
      popops::createSliceableTensor(graph, opts.dataType, opts.shape, {0}, {1},
                                    plan, sliceOptions, "embedding");
  } else {
    logging::info("Graph construction: create embedding matrix, grain size {}",
                  opts.grainSize);
    embeddingMatrix =
      popops::createSliceableTensor(graph, opts.dataType, opts.shape, {0}, {1},
                                    opts.grainSize, "embedding");
  }

  const auto rawEmbeddingMatrix =
      allocateHostMemoryForTensor(embeddingMatrix, "embeddingMatrix", graph,
                                  uploadProg, downloadProg, tmap);

  std::vector<unsigned> hostIndices(opts.numIndices);
  writeRandomValues(target, opts.indicesType, hostIndices, 0u,
                    static_cast<unsigned>(opts.shape->at(0) - 1), randomEngine);
  logging::trace("Indices: {}", hostIndices);

  const auto indices =
    createIndicesTensor(graph, {0}, opts.numIndices,
                        plan, sliceOptions, "offsets");
  const auto rawIndices =
      allocateHostMemoryForTensor(indices, "indices", graph,
                                  uploadProg, downloadProg, tmap);

  if (passEnabled(opts.pass, Pass::FWD)) {
    logging::info("Graph construction: create gather operation");
    const auto extractedData =
        popops::multiSlice(graph, embeddingMatrix, indices, {0}, {1}, prog,
                           plan, sliceOptions, "extracted");
    rawExtractedData =
        allocateHostMemoryForTensor(extractedData, "extractedData", graph,
                                    uploadProg, downloadProg, tmap);
  }

  if (passEnabled(opts.pass, Pass::WU)) {
    logging::info("Graph construction: create update operation");
    Tensor deltas;
    if (opts.useEmbeddingPlan) {
      deltas = popops::createSliceTensor(graph, opts.dataType, opts.shape,
                                         {0}, {1}, opts.numIndices,
                                         plan, sliceOptions, "deltas");
    } else {
      deltas = popops::createSliceTensor(graph, embeddingMatrix, {0}, {1},
                                         opts.numIndices, "deltas");
    }
    rawDeltas =
        allocateHostMemoryForTensor(deltas, "deltas", graph, uploadProg,
                                    downloadProg, tmap);

    const auto scale = graph.addConstant(opts.dataType, {}, opts.scale,
                                         "scale");
    graph.setTileMapping(scale, 0);

    popops::multiUpdateAdd(graph, embeddingMatrix, deltas, indices, scale, {0},
                          {1}, prog, plan, sliceOptions, "updated");
  }

  Sequence ctrlProg;
  if (!opts.ignoreData) {
    ctrlProg.add(uploadProg);
  }
  ctrlProg.add(prog);
  if (!opts.ignoreData) {
    ctrlProg.add(downloadProg);
  }

  logging::info("Create engine");
  Engine engine(graph, ctrlProg, {});

  const auto embeddingMatrixExtents =
      boost::extents[opts.shape->at(0)][opts.shape->at(1)];
  boost::multi_array<double, 2> hostEmbeddingMatrix(embeddingMatrixExtents);
  const auto extractedDataExtents =
      boost::extents[opts.numIndices][opts.shape->at(1)];
  boost::multi_array<double, 2> hostExtractedData(extractedDataExtents);
  boost::multi_array<double, 2> hostDeltas(extractedDataExtents);

  if (!opts.ignoreData) {
    logging::info("Generating the embedding matrix on the host");
    attachStreams(engine, tmap);

    writeRandomValues(target, opts.dataType, hostEmbeddingMatrix, -10., 10.,
                      randomEngine);
    copy(target, hostEmbeddingMatrix, opts.dataType, rawEmbeddingMatrix.get());
    copy(target, hostIndices.data(), hostIndices.size(), UNSIGNED_INT,
         rawIndices.get());

    if (passEnabled(opts.pass, Pass::WU)) {
      writeRandomValues(target, opts.dataType, hostDeltas, -1., 1.,
                        randomEngine);
      copy(target, hostDeltas, opts.dataType, rawDeltas.get());
    }
  }

  logging::info("Run program");
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  bool matchesModel = true;

  if (!opts.ignoreData) {
    const double absTol = opts.dataType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    const double relTol = opts.dataType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;

    boost::multi_array<double, 2> modelExtractedData(extractedDataExtents);
    boost::multi_array<double, 2> modelEmbeddingMatrix(embeddingMatrixExtents);

    if (passEnabled(opts.pass, Pass::FWD)) {
      logging::info("Validate gather operation against model");
      poplibs_test::embedding::multiSlice(hostEmbeddingMatrix, hostIndices,
                                          modelExtractedData);

      copy(target, opts.dataType, rawExtractedData.get(), hostExtractedData);
      matchesModel &= checkIsClose("multiSlice", hostExtractedData,
                                   modelExtractedData, relTol, absTol);
    }

    if (passEnabled(opts.pass, Pass::WU)) {
      logging::info("Validate update operation against model");
      std::copy_n(hostEmbeddingMatrix.data(),
                  hostEmbeddingMatrix.num_elements(),
                  modelEmbeddingMatrix.data());

      poplibs_test::embedding::multiUpdateAdd(hostDeltas, hostIndices,
                                              opts.scale, modelEmbeddingMatrix);

      copy(target, opts.dataType, rawEmbeddingMatrix.get(),
           hostEmbeddingMatrix);
      matchesModel &= checkIsClose("multiUpdateAdd", hostEmbeddingMatrix,
                                   modelEmbeddingMatrix, relTol, absTol);
    }
  }

  if (opts.profile) {
    engine.printProfileSummary(std::cout, {
      {"showExecutionSteps", opts.showExecutionSteps ? "true" : "false"},
      {"showVarStorage", opts.showVarStorage ? "true" : "false"},
    });
  }

  if (!matchesModel) {
    std::cerr << "Validation failed" << std::endl;
  }

  return matchesModel ? 0 : 1;
}
