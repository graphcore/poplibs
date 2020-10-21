// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/program_options.hpp>
#include <cassert>
#include <cstdint>
#include <poplar/CycleCount.hpp>
#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>
#include <vector>

#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplibs_support;
namespace po = boost::program_options;

namespace {

enum class CollectiveMethod {
  AUTO,
  CLOCKWISE_RING,
  ANTICLOCKWISE_RING,
  BIDIRECTIONAL_RING_PAIR,
  MEET_IN_MIDDLE_RING,
};

static const char *asString(CollectiveMethod method) {
  switch (method) {
  case CollectiveMethod::AUTO:
    return "auto";
  case CollectiveMethod::CLOCKWISE_RING:
    return "clockwise_ring";
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return "anticlockwise_ring";
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return "bidirectional_ring_pair";
  case CollectiveMethod::MEET_IN_MIDDLE_RING:
    return "meet_in_middle_ring";
  }
  throw poputil::poplibs_error("Unknown collective method");
}

static std::ostream &operator<<(std::ostream &os,
                                const CollectiveMethod &method) {
  os << asString(method);
  return os;
}

static std::istream &operator>>(std::istream &is, CollectiveMethod &method) {
  std::string token;
  is >> token;
  if (token == "auto")
    method = CollectiveMethod::AUTO;
  else if (token == "clockwise_ring")
    method = CollectiveMethod::CLOCKWISE_RING;
  else if (token == "anticlockwise_ring")
    method = CollectiveMethod::ANTICLOCKWISE_RING;
  else if (token == "bidirectional_ring_pair")
    method = CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
  else if (token == "meet_in_middle_ring")
    method = CollectiveMethod::MEET_IN_MIDDLE_RING;
  else
    throw poputil::poplibs_error("Unknown method <" + token + ">");
  return is;
}

} // End anonymous namespace

static Tensor createTensorToReduce(Graph &graph, const Type &type,
                                   unsigned numElements, bool shuffleMapping) {
  auto data = graph.addVariable(type, {numElements},
                                VariableMappingMethod::LINEAR, "data");
  if (shuffleMapping) {
    const auto numTiles = graph.getTarget().getNumTiles();
    // to check re ordering of collective is working shuffle tile mapping
    std::vector<std::vector<Interval>> m(graph.getTarget().getNumTiles());
    std::mt19937 g(0);
    for (unsigned e = 0; e < numElements; ++e) {
      auto t = g() % numTiles;
      m[t].push_back(Interval(e, e + 1));
    }
    graph.setTileMapping(data, m);
  }
  return data;
}

static Tensor createTensorToReduce(Graph &graph, const Type &type,
                                   unsigned numElements, bool shuffleMapping,
                                   const bool forceMapping,
                                   const unsigned forceIPU) {
  if (forceMapping) {
    const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
    auto forceGraph = graph.createVirtualGraph(forceIPU * tilesPerIPU,
                                               (forceIPU + 1) * tilesPerIPU);
    return createTensorToReduce(forceGraph, type, numElements, shuffleMapping);
  }
  return createTensorToReduce(graph, type, numElements, shuffleMapping);
}

static Tensor createOnIpuShuffled(Graph &graph, const Type &type,
                                  const Tensor &ref) {
  auto result = graph.addVariable(type, {ref.numElements()},
                                  VariableMappingMethod::LINEAR, "result");
  std::vector<std::vector<Interval>> m = graph.getTileMapping(ref);
  for (unsigned tile = 0; tile < graph.getTarget().getNumTiles(); tile += 2) {
    const auto swapTile = tile ^ 1;
    std::swap(m[tile], m[swapTile]);
  }
  graph.setTileMapping(result, m);
  return result;
}

enum class CollectiveOp { REDUCE_SCATTER, ALL_GATHER, ALL_REDUCE };

static const char *asString(CollectiveOp op) {
  switch (op) {
  case CollectiveOp::REDUCE_SCATTER:
    return "reduce_scatter";
  case CollectiveOp::ALL_GATHER:
    return "all_gather";
  case CollectiveOp::ALL_REDUCE:
    return "all_reduce";
  }
  throw poputil::poplibs_error("Unknown collective op");
}

static std::ostream &operator<<(std::ostream &os, const CollectiveOp &op) {
  os << asString(op);
  return os;
}

static std::istream &operator>>(std::istream &is, CollectiveOp &op) {
  std::string token;
  is >> token;
  if (token == "reduce_scatter")
    op = CollectiveOp::REDUCE_SCATTER;
  else if (token == "all_gather")
    op = CollectiveOp::ALL_GATHER;
  else if (token == "all_reduce")
    op = CollectiveOp::ALL_REDUCE;
  else
    throw poputil::poplibs_error("Unknown collective <" + token + ">");
  return is;
}

static double getOpInitialValue(popops::Operation op) {
  switch (op) {
  default:
    assert(0 && "Unexpected op");
  case popops::Operation::ADD:
    return 0.0;
  case popops::Operation::MUL:
    return 1.0;
  case popops::Operation::MIN:
    return std::numeric_limits<double>::infinity();
  case popops::Operation::MAX:
    return -std::numeric_limits<double>::infinity();
  }
}

/// Return the number of bytes sent over the links per byte of data.
static double getLinkBandwidthCorrectionFactor(CollectiveOp op,
                                               unsigned numIPUs) {
  auto n = static_cast<double>(numIPUs);
  switch (op) {
  case CollectiveOp::ALL_GATHER:
  case CollectiveOp::REDUCE_SCATTER:
    return (n - 1) / n;
  case CollectiveOp::ALL_REDUCE:
    return 2 * (n - 1) / n;
  }
  throw poputil::poplibs_error("Unknown collective op");
}

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  unsigned numIPUs = 4;
  boost::optional<unsigned> tilesPerIPU;
  unsigned numElements = 1024;
  unsigned ipusPerRank = 1;
  CollectiveOp collectiveOp = CollectiveOp::ALL_REDUCE;
  popops::Operation reduceOp = popops::Operation::ADD;
  CollectiveMethod collectiveMethod = CollectiveMethod::AUTO;
  bool replicateTopLevelGraph = false;
  bool shuffleMapping = false;
  unsigned iterations = 1;
  const auto type = poplar::HALF;
  // Some GCL config only support collectives on 1 side of the ladder
  // so add option to force the entire tensor onto single ipu
  bool forceMapping = false;
  bool inPlace = false;
  unsigned forceIpu = 0;
  // GCL only options.
  std::string maxBytesPerTile;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
    ("measure-overall-cycles", "Measure overall cycles")
    ("profile", "Output profiling report")
    ("use-replicated-implementation",
     "Use experimental replicated implementation")
    ("replicate-top-level-graph", po::value(&replicateTopLevelGraph),
     "Use a replicated top-level graph")
    ("collective", po::value(&collectiveOp)->default_value(collectiveOp),
     "Collective: reduce_scatter | all_gather | all_reduce")
    ("reduction-operator", po::value(&reduceOp)->default_value(reduceOp),
     "Reduction operator: ADD | MUL | MIN | MAX")
    ("elements", po::value(&numElements)->default_value(numElements),
     "Number of elements per rank")
    ("shuffle-mapping", po::value(&shuffleMapping)->default_value(false),
     "Shuffle the tile mapping of the input tensor")
    ("ipus-per-rank",
     po::value(&ipusPerRank)->default_value(ipusPerRank),
     "Number of IPUs in each rank")
    ("tiles-per-ipu", po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus", po::value(&numIPUs)->default_value(4),
     "Number of IPUs")
    ("in-place", po::value(&inPlace)->default_value(false),
      "Whether to do the operation in place")
    ("method",
     po::value(&collectiveMethod)->default_value(collectiveMethod),
     "Reduce method: auto | clockwise_ring | anticlockwise_ring | "
     "bidirectional_ring_pair | meet_in_middle_ring")
    ("force-mapping", po::value(&forceIpu),
         "for all elements onto one ipu")
    ("iterations,i",
     po::value(&iterations)->default_value(1),
     "Number of time the allReduce operation is called")
    ("gcl",
      "Set to use the GCL collectives implementation")
    ("gcl-max-bytes-per-tile", po::value<std::string>(&maxBytesPerTile),
      "The maximum number of bytes on one tile for one round of reduction");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (std::exception &e) {
    std::cerr << "error parsing command line: " << e.what() << "\n";
    return 1;
  }

  if (vm.count("help") != 0) {
    std::cout << desc << "\n\n";
    return 1;
  }

  if (iterations != 1 && inPlace) {
    std::cerr << "Can't operate in place for multiple iterations\n";
    return 1;
  }

  if (vm.count("force-mapping")) {
    forceMapping = true;
  }

  // Needed to set default arguments.
  po::notify(vm);

  switch (reduceOp) {
  case popops::Operation::ADD:
  case popops::Operation::MIN:
  case popops::Operation::MAX:
  case popops::Operation::MUL:
    break;
  default:
    std::cerr << "Unsupported reduction operator " << reduceOp << "\n";
    return 1;
  }

  auto device = [&]() -> TestDevice {
    if (isIpuModel(deviceType)) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      IPUModel ipuModel(deviceTypeToIPUName(deviceType));
      ipuModel.numIPUs = numIPUs;
      if (tilesPerIPU)
        ipuModel.tilesPerIPU = *tilesPerIPU;
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      if (tilesPerIPU)
        return createTestDevice(deviceType, numIPUs, *tilesPerIPU);
      else
        return createTestDeviceFullSize(deviceType, numIPUs);
    }
  }();
  auto replicationFactor = numIPUs / ipusPerRank;
  auto topLevelReplicationFactor =
      replicateTopLevelGraph ? replicationFactor : 1;
  Graph topLevelGraph(device.getTarget(),
                      replication_factor(topLevelReplicationFactor));
  popops::addCodelets(topLevelGraph);
  auto graph = topLevelGraph.createReplicatedGraph(replicationFactor /
                                                   topLevelReplicationFactor);
  Sequence uploadProg, downloadProg, prog;
  Tensor input, output;
  popops::Chunks reduceScatterOutput;
  if (collectiveOp != CollectiveOp::ALL_REDUCE) {
    std::cerr << "Collective operation \"" << collectiveOp
              << "\" is not yet supported\n";
    std::abort();
  }
  OptionFlags options = {{"method", asString(collectiveMethod)}};
  if (vm.count("use-replicated-implementation")) {
    options.set("useReplicatedImplementation", "true");
  }
  if (vm.count("gcl")) {
    std::cerr << "This version of replicated collectives is poplibs only\n";
    std::cerr
        << "To use --gcl, use the tool in gcl/tools/replicated_collectives\n";
    std::abort();
  }
  if (vm.count("gcl-max-bytes-per-tile")) {
    if (vm.count("gcl")) {
      options.set("maxBytesPerTile", maxBytesPerTile);
    } else {
      std::cerr << "gcl-max-bytes-per-tile only supported with --gcl\n";
      std::abort();
    }
  }

  input = createTensorToReduce(graph, type, numElements, shuffleMapping,
                               forceMapping, forceIpu);
  output = createOnIpuShuffled(graph, type, input);
  if (inPlace) {
    popops::replicatedAllReduceInPlace(graph, input, reduceOp, prog,
                                       "allReduce", options);
    // input gets zeroed before we read the output back in
    // allocateHostMemoryForTensor
    prog.add(Copy(input, output));
  } else {
    popops::replicatedAllReduceWithOutput(graph, input, output, reduceOp, prog,
                                          "allReduce", options);
  }
  bool doAllGather = collectiveOp == CollectiveOp::ALL_GATHER ||
                     collectiveOp == CollectiveOp::ALL_REDUCE;
  bool doReduceScatter = collectiveOp == CollectiveOp::REDUCE_SCATTER ||
                         collectiveOp == CollectiveOp::ALL_REDUCE;

  prog = program::Repeat(iterations, prog);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostInput = allocateHostMemoryForTensor(
      input, "input", graph, uploadProg, downloadProg, tmap);
  auto rawHostOutput = allocateHostMemoryForTensor(
      output, "output", graph, uploadProg, downloadProg, tmap);

  OptionFlags engineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.instrument", "true");
  }

  bool measureCycles = vm.count("measure-overall-cycles") &&
                       device.getTarget().getTargetType() == TargetType::IPU;
  Tensor cycleCount;
  std::unique_ptr<char[]> rawHostCycleCount;
  if (measureCycles) {
    cycleCount = poplar::cycleCount(topLevelGraph, prog, 0);
    rawHostCycleCount =
        allocateHostMemoryForTensor(cycleCount, "cycleCount", topLevelGraph,
                                    uploadProg, downloadProg, tmap);
  }
  Engine engine(topLevelGraph, {uploadProg, prog, downloadProg}, engineOptions);

  const auto numIpus = device.getTarget().getNumIPUs();
  const auto numPartials = numIpus / ipusPerRank;
  boost::multi_array<double, 1> hostChunks(boost::extents[numElements]);

  std::fill(hostChunks.data(), hostChunks.data() + hostChunks.num_elements(),
            getOpInitialValue(reduceOp));
  std::mt19937 randomEngine;
  const auto &target = device.getTarget();
  if (doReduceScatter) {
    boost::multi_array<double, 2> hostToReduce(
        boost::extents[numPartials][numElements]);
    double minimum = type == poplar::HALF ? 0.0 : -10.0;
    writeRandomValues(target, type, hostToReduce, minimum, +10.0, randomEngine);

    for (const auto &partial : hostToReduce) {
      for (unsigned i = 0; i != numElements; ++i) {
        switch (reduceOp) {
        default:
          assert(0 && "Unexpected op");
        case popops::Operation::ADD:
          hostChunks[i] += partial[i];
          break;
        case popops::Operation::MUL:
          hostChunks[i] *= partial[i];
          break;
        case popops::Operation::MIN:
          hostChunks[i] = std::min(hostChunks[i], partial[i]);
          break;
        case popops::Operation::MAX:
          hostChunks[i] = std::max(hostChunks[i], partial[i]);
          break;
        }
      }
    }
    copy(target, hostToReduce, type, rawHostInput.get());

  } else {
    assert(doAllGather);
    double minimum = type == poplar::HALF ? 0.0 : -10.0;
    writeRandomValues(target, type, hostChunks, minimum, +10.0, randomEngine);
    copy(target, hostChunks, type, rawHostInput.get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.disableExecutionProfiling();
    engine.run(0);
    engine.enableExecutionProfiling();
    engine.run(1);
    engine.disableExecutionProfiling();
    engine.run(2);
  });
  bool matchesModel;
  double relativeTolerance = type == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
  double absoluteTolerance = type == HALF ? HALF_ABS_TOL : FLOAT_ABS_TOL;
  if (doAllGather) {
    boost::multi_array<double, 2> hostGathered(
        boost::extents[numPartials][numElements]);
    copy(target, type, rawHostOutput.get(), hostGathered);
    boost::multi_array<double, 2> hostGatheredExpected(
        boost::extents[numPartials][numElements]);
    for (unsigned i = 0; i != numPartials; ++i) {
      hostGatheredExpected[i] = hostChunks;
    }
    matchesModel = checkIsClose("gathered", hostGathered, hostGatheredExpected,
                                relativeTolerance, absoluteTolerance);
  } else {
    boost::multi_array<double, 1> hostReduced(boost::extents[numElements]);
    copy(target, type, rawHostOutput.get(), hostReduced);
    matchesModel = checkIsClose("reduced", hostReduced, hostChunks,
                                relativeTolerance, absoluteTolerance);
  }
  if (vm.count("profile")) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }
  if (measureCycles) {
    std::uint64_t numCycles;
    std::memcpy(&numCycles, rawHostCycleCount.get(), sizeof(numCycles));
    std::cout << "Total cycle count: " << numCycles << "\n";
    const auto bytesPerIpu = numElements * (type == poplar::HALF ? 2 : 4);
    std::cout << "Bytes per IPU: " << bytesPerIpu << "\n";
    const auto bytesPerIpuPerCycle = (double)bytesPerIpu / numCycles;
    std::cout << "Algorithm bytes per IPU per cycle: " << bytesPerIpuPerCycle
              << "\n";
    const auto linkBytesPerIpuPerCycle =
        bytesPerIpuPerCycle *
        getLinkBandwidthCorrectionFactor(collectiveOp, numIpus);
    std::cout << "Link bytes per IPU per cycle: " << linkBytesPerIpuPerCycle
              << "\n";
  }
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  std::cerr << "Validation succeeded!\n";
  return 0;
}
