#include <cstdint>
#include <boost/program_options.hpp>
#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Collectives.hpp>
#include <popops/codelets.hpp>
#include <popsys/codelets.hpp>
#include <popsys/CycleCount.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poplibs_test/Util.hpp>
#include "TestDevice.hpp"
#include <cassert>
#include <vector>

#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
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
}

static std::ostream &
operator<<(std::ostream &os, const CollectiveMethod &method) {
  os << asString(method);
  return os;
}

static std::istream &
operator>>(std::istream &is, CollectiveMethod &method) {
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

// Return the IPUs in clockwise direction around the ring starting at IPU 0.
static std::vector<unsigned> getIpusInRing(int numIPUs) {
  std::vector<unsigned> ring;
  int ipu = 0;
  for (; ipu < numIPUs; ipu += 2) {
    ring.push_back(ipu);
  }
  ipu -= 1;
  if (ipu == numIPUs) {
    ipu -=2;
    assert(ipu < numIPUs);
  }
  for (; ipu >= 0; ipu -= 2) {
    ring.push_back(ipu);
  }
  return ring;
}

static Tensor
createTensorToReduce(Graph &graph, const Type &type, unsigned numElements,
                     bool shuffleMapping) {
  auto data = graph.addVariable(type, {numElements},
                                VariableMappingMethod::LINEAR,
                                "data");
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

static unsigned inverseRing(unsigned i, unsigned n) {
  if ((i & 1) == 0) {
    return i / 2;
  }
  return n - ((i + 1) / 2);
}

enum class CollectiveOp {
  REDUCE_SCATTER,
  ALL_GATHER,
  ALL_REDUCE
};

static const char *asString(CollectiveOp op) {
  switch (op) {
  case CollectiveOp::REDUCE_SCATTER:
    return "reduce_scatter";
  case CollectiveOp::ALL_GATHER:
    return "all_gather";
  case CollectiveOp::ALL_REDUCE:
    return "all_reduce";
  }
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

static Tensor concatChunks(popops::Chunks chunks) {
  std::sort(chunks.chunks.begin(), chunks.chunks.end(),
            [&] (popops::Chunk A, popops::Chunk B) {
    if (A.offset != B.offset) {
      return A.offset < B.offset;
    }
    return A.index < B.index;
  });
  std::vector<Tensor> toConcat(chunks.chunks.size());
  for (unsigned i = 0; i < chunks.chunks.size(); ++i) {
    toConcat[i] = chunks.chunks[i].tensor;
  }
  auto aa = concat(toConcat);
  return aa;
}

static double getOpInitialValue(popops::Operation op) {
  switch (op) {
  default: assert(0 && "Unexpected op");
  case popops::Operation::ADD: return 0.0;
  case popops::Operation::MUL: return 1.0;
  case popops::Operation::MIN: return std::numeric_limits<double>::infinity();
  case popops::Operation::MAX: return -std::numeric_limits<double>::infinity();
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
  case CollectiveOp::ALL_REDUCE: return 2 * (n - 1) / n;
  }
}

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  IPUModel ipuModel;
  ipuModel.numIPUs = 4;
  unsigned numElements = 1024;
  unsigned ipusPerRank = 1;
  CollectiveOp collectiveOp = CollectiveOp::ALL_REDUCE;
  popops::Operation reduceOp = popops::Operation::ADD;
  CollectiveMethod collectiveMethod = CollectiveMethod::AUTO;
  bool useReplicatedImplementation = false;
  bool shuffleMapping = false;
  const auto type = poplar::HALF;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
    ("measure-overall-cycles", "Measure overall cycles")
    ("profile", "Output profiling report")
    ("use-replicated-implementation",
     "Use experimental replicated implementation")
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
    ("tiles-per-ipu", po::value(&ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus", po::value(&ipuModel.numIPUs)->default_value(4),
     "Number of IPUs")
    ("method",
     po::value(&collectiveMethod)->default_value(collectiveMethod),
     "Reduce method: auto | clockwise_ring | anticlockwise_ring | "
     "bidirectional_ring_pair | meet_in_middle_ring");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (std::exception& e) {
    std::cerr << "error parsing command line: " << e.what() << "\n";
    return 1;
  }

  if (vm.count("help") != 0) {
    std::cout << desc << "\n\n";
    return 1;
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
    if (deviceType == DeviceType::IpuModel) {
      // When running on the IPU model we apply global exchange constraints,
      // which is why we create the device from the model here and not using
      // the normal createTestDevice factory function.
      addGlobalExchangeConstraints(ipuModel);
      setGlobalSyncLatency(ipuModel);
      return ipuModel.createDevice();
    } else {
      return createTestDevice(deviceType,
                              ipuModel.numIPUs,
                              ipuModel.tilesPerIPU);
    }
  }();

  Graph topLevelGraph(device.getTarget());
  auto replicationFactor = ipuModel.numIPUs / ipusPerRank;
  popops::addCodelets(topLevelGraph);
  popsys::addCodelets(topLevelGraph);
  auto graph = topLevelGraph.createReplicatedGraph(replicationFactor);
  Sequence uploadProg, downloadProg, prog;
  Tensor input, output;
  popops::Chunks reduceScatterOutput;
  if (collectiveOp != CollectiveOp::ALL_REDUCE) {
    std::cerr << "Collective operation \"" << collectiveOp
              << "\" is not yet supported\n";
    std::abort();
  }
  OptionFlags options = {
    {"method", asString(collectiveMethod)}
  };
  if (vm.count("use-replicated-implementation")) {
    options.set("useReplicatedImplementation", "true");
  }
  input = createTensorToReduce(graph, type, numElements, shuffleMapping);
  output =
      popops::replicatedAllReduce(graph, topLevelGraph, input, reduceOp,
                                  prog, "allReduce", options);
  bool doAllGather =
      collectiveOp == CollectiveOp::ALL_GATHER ||
      collectiveOp == CollectiveOp::ALL_REDUCE;
  bool doReduceScatter =
      collectiveOp == CollectiveOp::REDUCE_SCATTER ||
      collectiveOp == CollectiveOp::ALL_REDUCE;

  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostInput =
      allocateHostMemoryForTensor(topLevelGraph.getNonReplicatedTensor(input),
                                  "input", topLevelGraph, uploadProg,
                                  downloadProg, tmap);
  auto rawHostOutput =
      allocateHostMemoryForTensor(topLevelGraph.getNonReplicatedTensor(output),
                                  "output", topLevelGraph, uploadProg,
                                  downloadProg, tmap);

  OptionFlags engineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }

  bool measureCycles = vm.count("measure-overall-cycles") &&
                       device.getTarget().getTargetType() ==
                       TargetType::IPU;
  Tensor cycleCount;
  std::unique_ptr<char []> rawHostCycleCount;
  if (measureCycles) {
    cycleCount = popsys::cycleCount(topLevelGraph, prog, 0);
    rawHostCycleCount =
        allocateHostMemoryForTensor(cycleCount, "cycleCount", topLevelGraph,
                                    uploadProg, downloadProg, tmap);
  }
  Engine engine(topLevelGraph, {uploadProg, prog, downloadProg},
                engineOptions);

  const auto numIpus = device.getTarget().getNumIPUs();
  const auto numPartials = numIpus / ipusPerRank;
  boost::multi_array<double, 1>
      hostChunks(boost::extents[numElements]);

  std::fill(hostChunks.data(), hostChunks.data() + hostChunks.num_elements(),
            getOpInitialValue(reduceOp));
  std::mt19937 randomEngine;
  const auto &target = device.getTarget();
  if (doReduceScatter) {
    boost::multi_array<double, 2>
        hostToReduce(boost::extents[numPartials][numElements]);
    writeRandomValues(target, type, hostToReduce, -10.0, +10.0, randomEngine);

    for (const auto &partial : hostToReduce) {
      for (unsigned i = 0; i != numElements; ++i) {
        switch (reduceOp) {
        default: assert(0 && "Unexpected op");
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
    writeRandomValues(target, type, hostChunks, -10.0, +10.0, randomEngine);
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
    boost::multi_array<double, 2>
        hostGathered(boost::extents[numPartials][numElements]);
    copy(target, type, rawHostOutput.get(), hostGathered);
    boost::multi_array<double, 2>
        hostGatheredExpected(boost::extents[numPartials][numElements]);
    for (unsigned i = 0; i != numPartials; ++i) {
      hostGatheredExpected[i] = hostChunks;
    }
    matchesModel = checkIsClose("gathered",
                                hostGathered,
                                hostGatheredExpected,
                                relativeTolerance,
                                absoluteTolerance);
  } else {
    boost::multi_array<double, 1> hostReduced(boost::extents[numElements]);
    copy(target, type, rawHostOutput.get(), hostReduced);
    matchesModel = checkIsClose("reduced",
                                hostReduced,
                                hostChunks,
                                relativeTolerance,
                                absoluteTolerance);
  }
  if (vm.count("profile")) {
    engine.printProfileSummary(std::cout, OptionFlags{
      { "showExecutionSteps", "true" }
    });
  }
  if (measureCycles) {
    std::uint64_t numCycles;
    std::memcpy(&numCycles, rawHostCycleCount.get(), sizeof(numCycles));
    std::cout << "Total cycle count: " << numCycles << "\n";
    const auto bytesPerIpu = numElements * (type == poplar::HALF ? 2 : 4);
    std::cout << "Bytes per IPU: " << bytesPerIpu << "\n";
    const auto bytesPerIpuPerCycle = (double)bytesPerIpu / numCycles;
    std::cout << "Algorithm bytes per IPU per cycle: "
              << bytesPerIpuPerCycle << "\n";
    const auto linkBytesPerIpuPerCycle =
         bytesPerIpuPerCycle *
        getLinkBandwidthCorrectionFactor(collectiveOp, numIpus);
    std::cout << "Link bytes per IPU per cycle: "
              << linkBytesPerIpuPerCycle << "\n";
  }
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  std::cerr << "Validation succeeded!\n";
  return 0;
}