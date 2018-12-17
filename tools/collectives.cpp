#include <boost/program_options.hpp>
#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
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

struct Chunk {
  Tensor tensor;
  unsigned index;
};

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

static std::vector<Tensor>
splitIntoFragments(const Tensor &t, unsigned numFragments) {
  assert(t.rank() == 1);
  unsigned numElements = t.dim(0);
  // Fragments indexed by ipu.
  std::vector<Tensor> fragments;
  fragments.reserve(numFragments);
  for (unsigned fragment = 0; fragment != numFragments; ++fragment) {
    auto elementBegin = (numElements * fragment) / numFragments;
    auto elementEnd = std::min(numElements,
                               (numElements * (fragment + 1)) / numFragments);
    fragments.push_back(t.slice(elementBegin, elementEnd));
  }
  return fragments;
}

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

static void
ringReduceScatterStep(Graph &graph,
                      const std::vector<std::vector<Tensor>> &fragments,
                      const std::vector<unsigned> &ipuRing,
                      unsigned step,
                      std::vector<Tensor> &data,
                      std::vector<Tensor> &copySrcs,
                      std::vector<Tensor> &copyDsts,
                      std::vector<Tensor> &addOp0,
                      std::vector<Tensor> &addOp1,
                      bool clockwise) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto numFragments = fragments.size();
  assert(numFragments == numIpus);
  const auto numSteps = numFragments - 1;
  assert(step < numSteps);
  std::vector<Tensor> nextData(numIpus);
  for (unsigned i = 0; i != numIpus; ++i) {
    unsigned fragmentNum;
    unsigned recvIndex;
    if (clockwise) {
      // At each step the IPU at index N in the ring receives data from the
      // previous IPU in the ring and reduces it with a local fragment and
      // sends the data it reduced in the previous step to the next IPU in the
      // ring. We want the IPU at index N to have the reduced result for the
      // N'th fragment after the final step and so working backwards in step
      // (numSteps - 1 - x) the IPU at index (N - x) % numIpus should reduce
      // the data it receives with the N'th fragment. This can be rearranged
      // to give the following expression for the fragment that the IPU at
      // index i should reduce at each step:
      fragmentNum = (i + numSteps - 1 - step) % numIpus;
      recvIndex = (i + numIpus - 1) % numIpus;
    } else {
      // At each step the IPU at index N in the ring receives data from the
      // next IPU in the ring and reduces it with local fragment and sends
      // the data it reduced in the previous step to the previous IPU in the
      // ring. In step (numSteps - 1 - x) the IPU at index (N + x) % numIpus
      // should reduce the data it receives with the N'th fragment. Again this
      // can be rearranged to give the following expression for the fragment
      // that the IPU at index i should reduce at each step:
      fragmentNum = (i - (numSteps - 1 - step)) % numIpus;
      recvIndex = (i + 1) % numIpus;
    }
    auto ipu = ipuRing[i];
    auto recvIpu = ipuRing[recvIndex];
    auto copySrc = step == 0 ? fragments[recvIpu][fragmentNum] :
                               data[recvIndex];
    auto copyDst = poputil::cloneToIpu(graph, copySrc, ipu);
    copySrcs.push_back(copySrc);
    copyDsts.push_back(copyDst);
    addOp0.push_back(copyDst);
    addOp1.push_back(fragments[ipu][fragmentNum]);
    nextData[i] = copyDst;
  }
  data = nextData;
}

// Perform a collective reduce scatter operation.
static std::vector<Chunk>
unidirectionalRingReduceScatter(Graph &graph, Tensor toReduce,
                                Sequence &prog,
                                bool clockwise) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  assert(toReduce.dim(0) == numIpus);
  if (numIpus == 1) {
    return { {poputil::duplicate(graph, toReduce[0], prog), 0} };
  }
  const auto numFragments = numIpus;
  std::vector<std::vector<Tensor>> fragments;
  for (unsigned i = 0; i != numIpus; ++i) {
    fragments.push_back(splitIntoFragments(toReduce[i], numFragments));
  }

  auto ipuRing = getIpusInRing(numIpus);
  // Temporary data indexed by the position in the ring.
  std::vector<Tensor> data;
  const auto numSteps = ipuRing.size() - 1;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    std::vector<Tensor> addOp0;
    std::vector<Tensor> addOp1;
    ringReduceScatterStep(graph, fragments, ipuRing, step, data,
                          copySrcs, copyDsts, addOp0, addOp1, clockwise);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
    popops::addInPlace(graph, concat(addOp0), concat(addOp1), prog);
  }
  std::vector<Chunk> chunks(numIpus);
  for (unsigned i = 0; i != numIpus; ++i) {
    const auto ipu = ipuRing[i];
    chunks[ipu] = {data[i], i};
  }
  return chunks;
}

static std::vector<Chunk>
bidirectionalRingReduceScatter(Graph &graph, const Tensor &toReduce,
                               Sequence &prog) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  assert(toReduce.dim(0) == numIpus);
  if (numIpus == 1) {
    return { {poputil::duplicate(graph, toReduce[0], prog), 0} };
  }
  const auto numFragments = numIpus * 2;
  std::vector<std::vector<Tensor>> fragments;
  for (unsigned i = 0; i != numIpus; ++i) {
    fragments.push_back(splitIntoFragments(toReduce[i], numFragments));
  }
  // Split the fragments into two sets - even fragments are reduced
  // clockwise around the ring and odd fragments are reduced anticlockwise
  // around the ring.
  std::vector<std::vector<Tensor>> clockwiseFragments(numIpus);
  std::vector<std::vector<Tensor>> anticlockwiseFragments(numIpus);
  for (unsigned i = 0; i != numFragments; i +=2) {
    for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
      clockwiseFragments[ipu].push_back(fragments[ipu][i]);
      anticlockwiseFragments[ipu].push_back(fragments[ipu][i + 1]);
    }
  }

  auto ipuRing = getIpusInRing(numIpus);
  std::vector<Tensor> clockwiseData;
  std::vector<Tensor> anticlockwiseData;
  const auto numSteps = numIpus - 1;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    std::vector<Tensor> addOp0;
    std::vector<Tensor> addOp1;
    ringReduceScatterStep(graph, clockwiseFragments, ipuRing, step,
                          clockwiseData, copySrcs, copyDsts, addOp0, addOp1,
                          true);
    ringReduceScatterStep(graph, anticlockwiseFragments, ipuRing, step,
                          anticlockwiseData, copySrcs, copyDsts, addOp0, addOp1,
                          false);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
    popops::addInPlace(graph, concat(addOp0), concat(addOp1), prog);
  }
  std::vector<Tensor> data;
  for (unsigned i = 0; i != numIpus; ++i) {
    data.push_back(concat(clockwiseData[i], anticlockwiseData[i]));
  }
  std::vector<Chunk> chunks(numIpus);
  for (unsigned i = 0; i != numIpus; ++i) {
    const auto ipu = ipuRing[i];
    chunks[ipu] = {data[i], i};
  }
  return chunks;
}

static std::vector<Chunk>
reduceScatter(Graph &graph, const Tensor &toReduce, Sequence &prog,
              CollectiveMethod method) {
  if (method == CollectiveMethod::AUTO) {
    const auto numIPUs = graph.getTarget().getNumIPUs();
    method = numIPUs > 2 ? CollectiveMethod::BIDIRECTIONAL_RING :
                           CollectiveMethod::CLOCKWISE_RING;
  }
  switch (method) {
  default: assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING:
    return unidirectionalRingReduceScatter(graph, toReduce, prog, true);
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return unidirectionalRingReduceScatter(graph, toReduce, prog, false);
  case CollectiveMethod::BIDIRECTIONAL_RING:
    return bidirectionalRingReduceScatter(graph, toReduce, prog);
  }
}

static void
ringAllGatherStep(Graph &graph,
                  const std::vector<Chunk> &toGather,
                  const std::vector<unsigned> &ipuRing,
                  unsigned step,
                  std::vector<Chunk> &data,
                  std::vector<std::vector<Tensor>> &resultChunks,
                  std::vector<Tensor> &copySrcs,
                  std::vector<Tensor> &copyDsts,
                  bool clockwise) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  std::vector<Chunk> nextData(numIpus);
  for (unsigned i = 0; i != numIpus; ++i) {
    auto ipu = ipuRing[i];
    unsigned recvIndex;
    if (clockwise) {
      recvIndex = (i + numIpus - 1) % numIpus;
    } else {
      recvIndex = (i + 1) % numIpus;
    }
    auto &copySrc = step == 0 ? toGather[ipuRing[recvIndex]] :
                                data[recvIndex];
    auto copyDst = poputil::cloneToIpu(graph, copySrc.tensor, ipu);
    copySrcs.push_back(copySrc.tensor);
    copyDsts.push_back(copyDst);
    nextData[i] = { copyDst, copySrc.index };
    resultChunks[ipu][copySrc.index] = copyDst;
  }
  data = nextData;
}

static Tensor
unidirectionalRingAllGather(Graph &graph, const std::vector<Chunk> &toGather,
                            Sequence &prog, bool clockwise) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  assert(toGather.size() == numIpus);
  auto ipuRing = getIpusInRing(numIpus);
  std::vector<std::vector<Tensor>> resultChunks(numIpus,
                                                std::vector<Tensor>(numIpus));
  for (unsigned ipu = 0 ; ipu != numIpus; ++ipu) {
    resultChunks[ipu][toGather[ipu].index] =
        poputil::duplicate(graph, toGather[ipu].tensor, prog);
  }
  const auto numSteps = ipuRing.size() - 1;
  std::vector<Chunk> data;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    ringAllGatherStep(graph, toGather, ipuRing, step, data, resultChunks,
                      copySrcs, copyDsts, clockwise);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
  }
  std::vector<Tensor> result;
  result.reserve(numIpus);
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    result.push_back(concat(resultChunks[ipu]).expand({0}));
  }
  return concat(result);
}

static Tensor
bidirectionalRingAllGather(Graph &graph, const std::vector<Chunk> &toGather,
                           Sequence &prog) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  assert(toGather.size() == numIpus);
  auto ipuRing = getIpusInRing(numIpus);
  // Split the each chunk into two parts - one part that is sent clockwise
  // around the ring and one part that is sent anticlockwise around the
  // ring.
  std::vector<Chunk> clockwiseToGather;
  std::vector<Chunk> anticlockwiseToGather;
  std::vector<std::vector<Tensor>>
      clockwiseResultChunks(numIpus, std::vector<Tensor>(numIpus));
  std::vector<std::vector<Tensor>>
      anticlockwiseResultChunks(numIpus, std::vector<Tensor>(numIpus));
  for (unsigned ipu = 0 ; ipu != numIpus; ++ipu) {
    auto fragments = splitIntoFragments(toGather[ipu].tensor, 2);
    const auto index = toGather[ipu].index;
    clockwiseToGather.push_back({fragments[0], index});
    anticlockwiseToGather.push_back({fragments[1], index});
    clockwiseResultChunks[ipu][index] =
        poputil::duplicate(graph, fragments[0], prog);
    anticlockwiseResultChunks[ipu][index] =
        poputil::duplicate(graph, fragments[1], prog);
  }
  const auto numSteps = ipuRing.size() - 1;
  std::vector<Chunk> clockwiseData, anticlockwiseData;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    ringAllGatherStep(graph, clockwiseToGather, ipuRing, step, clockwiseData,
                      clockwiseResultChunks, copySrcs, copyDsts, true);
    ringAllGatherStep(graph, anticlockwiseToGather, ipuRing, step,
                      anticlockwiseData, anticlockwiseResultChunks, copySrcs,
                      copyDsts, false);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
  }
  std::vector<Tensor> result;
  result.reserve(numIpus);
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    std::vector<Tensor> resultChunks;
    for (unsigned chunk = 0 ; chunk != numIpus; ++chunk) {
      resultChunks.push_back(clockwiseResultChunks[ipu][chunk]);
      resultChunks.push_back(anticlockwiseResultChunks[ipu][chunk]);
    }
    result.push_back(concat(resultChunks).expand({0}));
  }
  return concat(result);
}

static Tensor
allGather(Graph &graph, const std::vector<Chunk> &toGather,
          Sequence &prog, CollectiveMethod method) {
  if (method == CollectiveMethod::AUTO) {
    const auto numIPUs = graph.getTarget().getNumIPUs();
    method = numIPUs > 2 ? CollectiveMethod::BIDIRECTIONAL_RING :
                           CollectiveMethod::CLOCKWISE_RING;
  }
  switch (method) {
  default: assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING:
    return unidirectionalRingAllGather(graph, toGather, prog, true);
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return unidirectionalRingAllGather(graph, toGather, prog, false);
  case CollectiveMethod::BIDIRECTIONAL_RING:
    return bidirectionalRingAllGather(graph, toGather, prog);
  }
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

static std::vector<Chunk>
createChunksToGather(Graph &graph, const Type &type, unsigned numElements) {
  // Order the chunks by the index of the IPU in the ring to match the
  // output of the reduce scatter collective.
  const auto numIpus = graph.getTarget().getNumIPUs();
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  auto ipuRing = getIpusInRing(numIpus);
  std::vector<Chunk> chunks(numIpus);
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

static Tensor concatChunks(std::vector<Chunk> chunks) {
  const auto numChunks = chunks.size();
  std::vector<Tensor> toConcat(numChunks);
  for (auto &chunk : chunks) {
    assert(chunk.index < numChunks);
    toConcat[chunk.index] = chunk.tensor;
  }
  return concat(toConcat);
}

const auto type = poplar::HALF;

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  unsigned tilesPerIPU = IPUModel().tilesPerIPU;
  unsigned numIPUs = 4;
  unsigned numElements = 1024;
  CollectiveOp collectiveOp = CollectiveOp::ALL_REDUCE;
  CollectiveMethod collectiveMethod = CollectiveMethod::AUTO;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("collective", po::value(&collectiveOp)->default_value(collectiveOp),
     "Collective: reduce_scatter | all_gather | all_reduce")
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

  auto device = createTestDevice(deviceType, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  Sequence uploadProg, downloadProg, prog;
  Tensor input, output;
  std::vector<Chunk> reduceScatterOutput;
  bool doReduceScatter =
      collectiveOp == CollectiveOp::REDUCE_SCATTER |
      collectiveOp == CollectiveOp::ALL_REDUCE;
  if (doReduceScatter) {
    input = createTensorToReduce(graph, type, numElements);
    reduceScatterOutput = reduceScatter(graph, input, prog, collectiveMethod);
    output = concatChunks(reduceScatterOutput);
  }
  bool doAllGather =
      collectiveOp == CollectiveOp::ALL_GATHER |
      collectiveOp == CollectiveOp::ALL_REDUCE;
  if (doAllGather) {
    std::vector<Chunk> allGatherInput;
    if (doReduceScatter) {
      allGatherInput = reduceScatterOutput;
    } else {
      allGatherInput = createChunksToGather(graph, type, numElements);
      input = concatChunks(allGatherInput);
    }
    output = allGather(graph, allGatherInput, prog, collectiveMethod);
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
  std::mt19937 randomEngine;
  const auto &target = device.getTarget();
  if (doReduceScatter) {
    boost::multi_array<double, 2>
        hostToReduce(boost::extents[numIpus][numElements]);
    writeRandomValues(target, type, hostToReduce, -10.0, +10.0, randomEngine);
    for (const auto &partial : hostToReduce) {
      for (unsigned i = 0; i != numElements; ++i) {
        hostChunks[i] += partial[i];
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
