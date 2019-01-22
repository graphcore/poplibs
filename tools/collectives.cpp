#include <boost/program_options.hpp>
#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Collectives.hpp>
#include <popops/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poplibs_test/Util.hpp>
#include "TestDevice.hpp"
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
  BIDIRECTIONAL_RING
};

static const char *asString(CollectiveMethod method) {
  switch (method) {
  case CollectiveMethod::AUTO:
    return "auto";
  case CollectiveMethod::CLOCKWISE_RING:
    return "clockwise_ring";
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return "anticlockwise_ring";
  case CollectiveMethod::BIDIRECTIONAL_RING:
    return "bidirectional_ring";
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
  else if (token == "bidirectional_ring")
    method = CollectiveMethod::BIDIRECTIONAL_RING;
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
createTensorToReduce(Graph &graph, const Type &type, unsigned numElements) {
  std::vector<Tensor> toReduceVec;
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    auto ipuGraph = graph.createVirtualGraph(ipu * tilesPerIpu,
                                             (ipu + 1) * tilesPerIpu);
    auto data = ipuGraph.addVariable(type, {numElements},
                                     VariableMappingMethod::LINEAR,
                                     "data" + std::to_string(ipu));

    toReduceVec.push_back(data.expand({0}));
  }
  auto toReduce = concat(toReduceVec);
  return toReduce;
}

static std::vector<popops::Chunk>
createChunksToGather(Graph &graph, const Type &type, unsigned numElements) {
  // Order the chunks by the index of the IPU in the ring to match the
  // output of the reduce scatter collective.
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  auto ipuRing = getIpusInRing(numIpus);
  std::vector<popops::Chunk> chunks(numIpus);
  for (unsigned i = 0; i != numIpus; ++i) {
    const auto ipu = ipuRing[i];
    const auto elementBegin = (numElements * i) / numIpus;
    const auto elementEnd = (numElements * (i + 1)) / numIpus;
    auto ipuGraph = graph.createVirtualGraph(ipu * tilesPerIpu,
                                             (ipu + 1) * tilesPerIpu);
    auto data = ipuGraph.addVariable(type, {elementEnd - elementBegin},
                                     VariableMappingMethod::LINEAR,
                                     "data" + std::to_string(ipu));
    chunks[ipu] = { data, i };
  }
  return chunks;
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

static Tensor concatChunks(std::vector<popops::Chunk> chunks) {
  const auto numChunks = chunks.size();
  std::vector<Tensor> toConcat(numChunks);
  for (auto &chunk : chunks) {
    assert(chunk.index < numChunks);
    toConcat[chunk.index] = chunk.tensor;
  }
  return concat(toConcat);
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

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  unsigned tilesPerIPU = IPUModel().tilesPerIPU;
  unsigned numIPUs = 4;
  unsigned numElements = 1024;
  CollectiveOp collectiveOp = CollectiveOp::ALL_REDUCE;
  popops::Operation reduceOp = popops::Operation::ADD;
  CollectiveMethod collectiveMethod = CollectiveMethod::AUTO;
  const auto type = poplar::HALF;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("collective", po::value(&collectiveOp)->default_value(collectiveOp),
     "Collective: reduce_scatter | all_gather | all_reduce")
    ("reduction-operator", po::value(&reduceOp)->default_value(reduceOp),
     "Reduction operator: ADD | MUL | MIN | MAX")
    ("elements", po::value(&numElements)->default_value(numElements),
     "Number of elements per IPU")
    ("tiles-per-ipu", po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus", po::value(&numIPUs)->default_value(4),
     "Number of IPUs")
    ("method",
     po::value(&collectiveMethod)->default_value(collectiveMethod),
     "Reduce method: auto | clockwise_ring | anticlockwise_ring | "
     "bidirectional_ring");

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

  auto device = createTestDevice(deviceType, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Sequence uploadProg, downloadProg, prog;
  Tensor input, output;
  std::vector<popops::Chunk> reduceScatterOutput;
  bool doReduceScatter =
      collectiveOp == CollectiveOp::REDUCE_SCATTER |
      collectiveOp == CollectiveOp::ALL_REDUCE;
  if (doReduceScatter) {
    input = createTensorToReduce(graph, type, numElements);
    reduceScatterOutput =
        popops::reduceScatter(graph, input, reduceOp, prog,
                              "reduceScatter",
                              {{"method", asString(collectiveMethod)}});
    output = concatChunks(reduceScatterOutput);
  }
  bool doAllGather =
      collectiveOp == CollectiveOp::ALL_GATHER |
      collectiveOp == CollectiveOp::ALL_REDUCE;
  if (doAllGather) {
    std::vector<popops::Chunk> allGatherInput;
    if (doReduceScatter) {
      allGatherInput = reduceScatterOutput;
    } else {
      allGatherInput = createChunksToGather(graph, type, numElements);
      input = concatChunks(allGatherInput);
    }
    output =
        popops::allGather(graph, allGatherInput, prog, "allGather",
                          {{"method", asString(collectiveMethod)}});
  }

  std::vector<std::pair<std::string, char*>> tmap;
  auto rawHostInput =
      allocateHostMemoryForTensor(input, "input", graph,
                                  uploadProg, downloadProg, tmap);
  auto rawHostOutput =
      allocateHostMemoryForTensor(output, "output", graph, uploadProg,
                                  downloadProg, tmap);
  OptionFlags engineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);

  const auto numIpus = device.getTarget().getNumIPUs();
  boost::multi_array<double, 1>
      hostChunks(boost::extents[numElements]);

  std::fill(hostChunks.data(), hostChunks.data() + hostChunks.num_elements(),
            getOpInitialValue(reduceOp));
  std::mt19937 randomEngine;
  const auto &target = device.getTarget();
  if (doReduceScatter) {
    boost::multi_array<double, 2>
        hostToReduce(boost::extents[numIpus][numElements]);
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
    engine.run(0);
  });
  bool matchesModel;
  double relativeTolerance = type == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
  double absoluteTolerance = type == HALF ? HALF_ABS_TOL : FLOAT_ABS_TOL;
  if (doAllGather) {
    boost::multi_array<double, 2>
        hostGathered(boost::extents[numIpus][numElements]);
    copy(target, type, rawHostOutput.get(), hostGathered);
    boost::multi_array<double, 2>
        hostGatheredExpected(boost::extents[numIpus][numElements]);
    for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
      hostGatheredExpected[ipu] = hostChunks;
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
    engine.printSummary(std::cout, OptionFlags{
      { "doLayerWiseBreakdown", "true" }
    });
  }
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  std::cerr << "Validation succeeded!\n";
  return 0;
}
