#include "TestDevice.hpp"
#include <poplar/Engine.hpp>

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_test/Util.hpp"
#include "popnn/codelets.hpp"
#include "poputil/VertexTemplates.hpp"

#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;

template <typename DataType>
std::vector<std::vector<DataType>> cpp_model(std::vector<DataType> in,
                                             unsigned topK, unsigned size,
                                             unsigned divisorLog2) {
  const unsigned divisor = (1u << divisorLog2);
  const int nOutputs = (size + divisor - 1) / divisor;
  std::vector<std::vector<DataType>> modelMaxActs;

  for (int i = 0; i < nOutputs; ++i) {
    modelMaxActs.push_back({});
    std::vector<DataType> &v = modelMaxActs.back();

    unsigned maxI = divisor * i;
    const unsigned end = (maxI + divisor > size) ? size : maxI + divisor;
    for (std::size_t j = maxI; j < end; ++j) {
      v.push_back(in[j]);
    }

    while (v.size() < topK) {
      v.push_back(std::numeric_limits<DataType>::lowest());
    }
    std::sort(v.begin(), v.end(), std::greater<DataType>());
    v.erase(v.begin() + topK, v.end());
  }

  return modelMaxActs;
}

template <typename DataType>
static bool doTest(const DeviceType &deviceType, const Type &dataType,
                   const Type &labelType, unsigned divisor, unsigned size,
                   unsigned topK, bool sort) {
  auto device = createTestDevice(deviceType);
  auto &target = device.getTarget();

  const short unsigned int divisorLog2 = poplibs_support::ceilLog2(divisor);
  divisor = (1u << divisorLog2);

  const auto nOutputs = (size + divisor - 1) / divisor;

  Graph graph(target);
  popnn::addCodelets(graph);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;

  // Allocate the activations, a.k.a the inputs.
  auto activations = graph.addVariable(dataType, {size}, "activations");
  graph.setTileMapping(activations, 0);
  auto rawHostActivations = allocateHostMemoryForTensor(
      activations, "activations", graph, uploadProg, downloadProg, tmap);

  // Allocate the input indices.
  auto maxIndices =
      graph.addVariable(UNSIGNED_INT, {nOutputs * topK}, "maxValuesIndices");
  graph.setTileMapping(maxIndices, 0);
  auto rawHostMaxIndices = allocateHostMemoryForTensor(
      maxIndices, "maxValuesIndices", graph, uploadProg, downloadProg, tmap);

  // Allocate the partial values.
  auto partialValues =
      graph.addVariable(dataType, {nOutputs * topK}, "maxValuePartials");
  graph.setTileMapping(partialValues, 0);
  auto rawModelPartialValues = allocateHostMemoryForTensor(
      partialValues, "maxValuePartials", graph, uploadProg, downloadProg, tmap);

  std::mt19937 randomEngine;

  std::vector<DataType> hostActivations(size);
  boost::random::uniform_int_distribution<unsigned> randDist(
      std::numeric_limits<unsigned>::lowest(),
      std::numeric_limits<unsigned>::max());

  for (auto &a : hostActivations) {
    unsigned tmp = randDist(randomEngine);
    a = *reinterpret_cast<DataType *>(&tmp);

    // Remove NANs.
    if (std::isnan(a)) {
      tmp = tmp >> 2;
      a = *reinterpret_cast<DataType *>(&tmp);
    }

    // Flush denormals to zero.
    if (std::is_floating_point<DataType>::value &&
        std::fabs(a) < std::numeric_limits<DataType>::min()) {
      a = 0;
    }
  }
  copy(target, hostActivations.data(), size, dataType,
       rawHostActivations.get());

  auto cs = graph.addComputeSet();

  std::string vertexName =
      templateVertex("popnn::ReduceMaxNClassGather", dataType, sort);

  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, 0);

  unsigned index = 0;
  graph.connect(v["activations"], activations);
  graph.connect(v["maxValues"], partialValues);
  graph.connect(v["maxValuesIndices"], maxIndices);
  graph.setInitialValue(v["index"], index);
  graph.setInitialValue(v["size"], size);
  graph.setInitialValue(v["divisorLog2"], divisorLog2);
  graph.setInitialValue(v["numK"], topK);
  graph.setInitialValue(v["shouldSort"], sort);

  Engine e(std::move(graph), Sequence(uploadProg, Execute(cs), downloadProg));
  attachStreams(e, tmap);

  device.bind([&](const Device &d) { e.loadAndRun(d); });

  // Create a flattened array of outputs from the device.
  std::vector<DataType> flattened_device_array;
  flattened_device_array.resize(nOutputs * topK);

  copy(target, dataType, rawModelPartialValues.get(),
       flattened_device_array.data(), nOutputs * topK);

  std::vector<unsigned> flattened_index_array(nOutputs * topK);
  copy(target, labelType, rawHostMaxIndices.get(), flattened_index_array.data(),
       nOutputs * topK);

  // Run the C++ model.
  std::vector<std::vector<DataType>> cpp_out =
      cpp_model(hostActivations, topK, size, divisorLog2);

  std::vector<DataType> flattened_cpp_array;
  for (std::vector<DataType> &topk : cpp_out) {
    for (DataType val : topk) {
      flattened_cpp_array.push_back(val);
    }
  }
  bool success = true;
  // We check the indices before we do any sorting stuff.
  for (int i = 0; i < nOutputs * topK; ++i) {
    unsigned ind = flattened_index_array[i];
    if (ind != std::numeric_limits<unsigned>::max()) {
      success &= hostActivations[ind] == flattened_device_array[i];
    }
  }

  // We have to carefuly sort the flattened device output by sorting each topK
  // subarray separately. We do this if we aren't testing the inbuilt sorting
  // mechanism in the vertex.
  if (!sort) {
    for (int i = 0; i < nOutputs; ++i) {
      std::sort(flattened_device_array.begin() + i * topK,
                flattened_device_array.begin() + ((i + 1) * topK),
                std::greater<DataType>());
    }
  }

  // Tolerance is zero as we are only copying the data around, should be no
  // floating point error.`
  for (int i = 0; i < nOutputs * topK; ++i) {
    success &= flattened_cpp_array[i] == flattened_device_array[i];
  }
  return success;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type activationType;
  Type labelType = UNSIGNED_INT;
  unsigned divisor, size, topK;
  po::options_description desc("Options");
  desc.add_options()("help", "Print help")(
      "device-type", po::value<DeviceType>(&deviceType)->required(),
      "Device Type")("activation-type",
                     po::value<Type>(&activationType)->required(),
                     "Element type for input activations")(
      "divisor", po::value<unsigned>(&divisor)->required(),
      "Factor by which to reduce max class")(
      "size", po::value<unsigned>(&size)->required(),
      "Total size to process with vertex")(
      "k", po::value<unsigned>(&topK)->required(),
      "Find the 'k' amount of top elements in the set");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (activationType == FLOAT) {
    // Test without sorting the output.
    if (!doTest<float>(deviceType, activationType, labelType, divisor, size,
                       topK, false))
      return 1;

    // Check with sorting the output.
    if (!doTest<float>(deviceType, activationType, labelType, divisor, size,
                       topK, true))
      return 1;
  } else if (activationType == INT) {
    // Test without sorting the output.
    if (!doTest<int>(deviceType, activationType, labelType, divisor, size, topK,
                     false))
      return 1;

    // Check with sorting the output.
    if (!doTest<int>(deviceType, activationType, labelType, divisor, size, topK,
                     true))
      return 1;
  } else if (activationType == UNSIGNED_INT) {
    // Test without sorting the output.
    if (!doTest<unsigned>(deviceType, activationType, labelType, divisor, size,
                          topK, false))
      return 1;

    // Check with sorting the output.
    if (!doTest<unsigned>(deviceType, activationType, labelType, divisor, size,
                          topK, true))
      return 1;
  } else {
    // Type is unsupported.
    return 1;
  }
  return 0;
}
