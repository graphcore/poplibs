#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <random>

#include "TestDevice.hpp"
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_test/Reduce.hpp>
#include <poplibs_test/Util.hpp>

#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/random.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_test::util;
namespace po = boost::program_options;
namespace br = boost::random;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#define MAX_TILES_TO_USE_SIM_TARGET 20
#define MAX_TILES_TO_USE_DEFAULT 64
#define MAX_IPUS_TO_USE 1

const OptionFlags defaultEngineOptions{
    {"target.workerStackSizeInBytes", "0x200"}};

// Split a string and call f(part) for each split.
template <typename StringFunction>
void splitString(const std::string &str, char delimiter, StringFunction f) {
  std::size_t from = 0;
  for (std::size_t i = 0; i < str.size(); ++i) {
    if (str[i] == delimiter) {
      f(str.substr(from, i - from));
      from = i + 1;
    }
  }
  if (from <= str.size())
    f(str.substr(from, str.size() - from));
}

// Read a vector like "1,2,3". Spaces are not allowed, although stoul will
// allow them preceding a number so "1, 2, 3" is ok but "1 ,2 ,3" will fail.
//
// Throws an exception on failure.
template <class vecType>
std::vector<vecType> parseSizeVector(const std::string &token) {
  std::vector<vecType> vec;

  if (token.empty())
    return vec;

  splitString(token, ',', [&](const std::string &part) {
    // stoul throws an exception if there's no number but you also need
    // to check if there are trailing non-number chars, e.g.
    // "12why_doesnt_cpp_have_a_sane_integer_parsing_function"
    std::size_t idx = 0;
    vec.push_back(std::stoul(part, &idx));
    if (idx != part.size()) {
      throw poputil::poplibs_error("Invalid integer <" + part + ">");
    }
  });

  return vec;
}

// Generate a random tensor shape, with some dimensions randomly set to
// 1 because it is an edge case.
std::vector<std::size_t> getRandomShape(std::mt19937 &gen, unsigned tiles) {
  auto rank = br::uniform_int_distribution<>(1, 5)(gen);

  // Distribution over the number of elements.
  const auto numElems = rank * 10000 / tiles;
  auto expectedNumel = br::binomial_distribution<>(numElems)(gen) + 1;

  // Distribution over the dimensions.
  br::uniform_int_distribution<> dimDist(1, pow(expectedNumel, 1.0 / rank) * 2);

  // Probability of setting a dimension to 1.
  br::bernoulli_distribution<double> dimOneDist(0.05);

  std::vector<std::size_t> shape(rank);

  for (auto &s : shape) {
    if (dimOneDist(gen))
      s = 1;
    else
      s = dimDist(gen);
  }

  return shape;
}

// Get random dimensions to reduce for a given shape.
std::vector<std::size_t> getRandomDims(std::mt19937 &gen, std::size_t rank) {

  // Generate a distribution over the number of dimensions to reduce. I expect
  // that reducing 0 or all dimensions is uncommon so give them lower priority.

  std::vector<double> weights;
  weights.push_back(1.0); // 0: don't reduce any dimensions.
  for (std::size_t i = 2; i < rank; ++i)
    weights.push_back(10.0); // 1..size()-1: reduce some but not all dimensions.
  if (rank > 0)
    weights.push_back(1.0); // Reduce all dimensions.

  br::discrete_distribution<> numDimsToReduceDist(weights.begin(),
                                                  weights.end());
  auto numDimsToReduce = numDimsToReduceDist(gen);

  std::vector<bool> reduceDim(rank);
  for (int i = 0; i < numDimsToReduce; ++i)
    reduceDim[i] = true;

  // Shuffle, so random dimensions are chosen.
  std::shuffle(reduceDim.begin(), reduceDim.end(), gen);

  std::vector<std::size_t> dims;

  for (std::size_t i = 0; i < reduceDim.size(); ++i)
    if (reduceDim[i])
      dims.push_back(i);

  // Shuffle the dims in case some code only works when they're ordered.
  std::shuffle(dims.begin(), dims.end(), gen);

  return dims;
}

// Get a random operation.
popops::Operation getRandomOp(std::mt19937 &gen) {
  // Randomly choose an op from ADD, SQUARE_ADD, MUL, MIN, MAX, AND and OR.
  return static_cast<popops::Operation>(
      br::uniform_int_distribution<>(0, 6)(gen));
}

// Get a random scale - normally distributed with mean and stddev 1.
float getRandomScale(std::mt19937 &gen, popops::Operation op) {
  // Only (square)add can scale.
  if (op != popops::Operation::ADD && op != popops::Operation::SQUARE_ADD)
    return 1.0f;
  return br::normal_distribution<>(-2.0f, 2.0f)(gen);
}

// Random decision on whether or not to do an update operation.
bool getRandomUpdate(std::mt19937 &gen, popops::Operation op) {
  // Only (square)add can update.
  if (op != popops::Operation::ADD && op != popops::Operation::SQUARE_ADD)
    return false;
  return br::bernoulli_distribution<double>(0.3)(gen);
}

// Randomly decide between reduce() and reduceWithOutput().
bool getRandomWithOutput(std::mt19937 &gen) {
  return br::bernoulli_distribution<double>(0.5)(gen);
}

// Get a randm number of IPUs.
unsigned getRandomNumIPUs(std::mt19937 &gen) {
  return br::uniform_int_distribution<>(1, MAX_IPUS_TO_USE)(gen);
}

// And a random number of tiles.
unsigned getRandomTilesPerIPU(std::mt19937 &gen, unsigned maxTiles) {
  return 4 * br::uniform_int_distribution<>(1, maxTiles / 4)(gen);
}

// Randomly decide between the Sequence-based API (false) and the
// vector<ComputeSet> one (true).
bool getRandomApi(std::mt19937 &gen) {
  return br::bernoulli_distribution<double>(0.5)(gen);
}

// Get random input and output types. This is only used when the operation
// isnt AND or OR - in that case they have to be BOOL.
poplar::Type getRandomTypes(std::mt19937 &gen, popops::Operation op) {
  if (op == popops::Operation::LOGICAL_AND ||
      op == popops::Operation::LOGICAL_OR) {
    return BOOL;
  }

  switch (br::uniform_int_distribution<>(0, 2)(gen)) {
  case 0:
    return HALF;
  case 1:
    return FLOAT;
  case 2:
    return INT;
  }

  POPLIB_UNREACHABLE();
}

bool validateParameters(const po::variables_map &vm) {
  // If a file is specified you cannot set the shape.
  if (vm.count("file") != 0) {
    if (vm.count("shape") != 0) {
      std::cerr << "--shape cannot be used with --file\n";
      return false;
    }
  }

  // If a file or seed is not specified you must set the shape.
  if (vm.count("file") == 0 && vm.count("seed") == 0) {
    if (vm.count("shape") == 0) {
      std::cerr << "--shape must be specified (or use --seed or --file)\n";
      return false;
    }
  }

  // If a seed is not specified you must set the dimensions to reduce.
  if (vm.count("seed") == 0) {
    if (vm.count("dims") == 0) {
      std::cerr << "--dims must be specified (or use --seed)\n";
      return false;
    }
  }

  return true;
}

std::vector<std::size_t> getReducedShape(const std::vector<std::size_t> &shape,
                                         const std::vector<std::size_t> &dims) {
  // Is the given dimension one that should be reduced.
  auto isReducedDim = [&](std::size_t dim) {
    return std::find(dims.begin(), dims.end(), dim) != dims.end();
  };

  std::vector<std::size_t> reducedShape;

  for (unsigned i = 0; i < shape.size(); ++i) {
    if (!isReducedDim(i)) {
      reducedShape.push_back(shape[i]);
    }
  }
  return reducedShape;
}

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  // Defaul input parameters.
  Type dataType = FLOAT;
  popops::Operation op = popops::Operation::ADD;
  float scale = 1.0f;
  bool update = false;
  bool withOutput = false;
  bool computeSetApi = false;
  int seed = 0;
  std::string initialShapeString;
  std::string shuffleString;
  std::string shapeString;
  std::string dimsString;
  std::string file;
  std::vector<std::size_t> initialShape;
  std::vector<unsigned> shuffle;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> dims;

  IPUModel ipuModel;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("seed", po::value(&seed),
      "Do a random reduction with the given seed. No other options "
      "are required, if they are specified they override the randomly chosen"
      "settings.")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report")
    ("file", po::value(&file),
      "If specified, load the input and optionally output tensors from "
      "a file. The file must be a binary serialisation of the tensors "
      "with at least one tensor. The first tensor is the reduction input.")
    ("type", po::value(&dataType),
      "Data type of input and output values (half, float, int, bool, etc.).")
    ("operation", po::value(&op),
      "The operation to perform (ADD, SQUARE_ADD, MUL, MIN, MAX,"
                                 " LOGICAL_AND or LOGICAL_OR)")
    ("scale", po::value(&scale),
      "Scale the final value by this amount.")
    ("update", po::value(&update),
      "If true, do `out += reduce(in)`, otherwise do `out = reduce(in)`")
    ("withoutput", po::value(&withOutput),
      "If true use reduceWithOutput(), otherwise use reduce()")
    ("computesetapi", po::value(&computeSetApi),
      "If true use the vector<ComputeSet> API instead of the Sequence one.")
    ("shape", po::value(&shapeString),
      "The shape of the input tensor, e.g. `4,2,3`")
    ("dims", po::value(&dimsString),
      "The dimensions to reduce, e.g. `1,0,2`")
    ("initial-shape", po::value(&initialShapeString),
      "The shape of the input tensor when created, e.g. `2,2,2,3`")
    ("shuffle", po::value(&shuffleString),
      "Dim shuffle to apply to the input tensor with shape initial-shape."
      " e.g. 1,0,2,3")
    ("tiles-per-ipu", po::value(&ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus", po::value(&ipuModel.numIPUs),
     "Number of IPUs");
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  } catch (std::exception &e) {
    std::cerr << "error parsing command line: " << e.what() << "\n";
    return 1;
  }

  if (vm.count("help") != 0) {
    std::cerr << "This tool performs a reduction operation. You can explicitly "
                 "set the\n"
                 "data type, operation, shape and so on using the options "
                 "below. If any\n"
                 "are not specified, and --seed is used then they will be set "
                 "randomly.\n"
                 "If --file is used then the shape, tile mapping and type will "
                 "be loaded\n"
                 "from the file.\n\n"
                 "";

    std::cerr << desc << "\n";
    return 1;
  }

  // Needed to set default arguments.
  po::notify(vm);

  // Validate that incompatible parameters are not given up-front for
  // speed.
  if (!validateParameters(vm))
    return 1;

  // Initialise the seed.
  std::mt19937 randomEngine;
  randomEngine.seed(seed);

  // Set the random model parameters if --seed was specified and they
  // weren't overridden with --tiles-per-ipu or --ipus.
  if (vm.count("seed") != 0) {
    if (vm.count("tiles-per-ipu") == 0) {
      std::cerr << "Randomly setting tiles-per-ipu.\n";
      const unsigned maxTiles = deviceType == DeviceType::Sim
                                    ? MAX_TILES_TO_USE_SIM_TARGET
                                    : MAX_TILES_TO_USE_DEFAULT;
      ipuModel.tilesPerIPU = getRandomTilesPerIPU(randomEngine, maxTiles);
    }
    if (vm.count("ipus") == 0) {
      std::cerr << "Randomly setting ipus.\n";
      ipuModel.numIPUs = getRandomNumIPUs(randomEngine);
    }
  }

  std::cerr << "Initializing graph...\n";

  auto device =
      createTestDevice(deviceType, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  Tensor input;

  // Load the tensors if --file was specified.
  if (vm.count("file") != 0) {
    std::cerr << "Loading tensors from '" << file << "'\n";

    std::ifstream in(file);
    auto tensors = graph.deserializeTensors(in, SerializationFormat::Binary);
    // TODO: T12989 When there are two tensors we can use one as the input and
    // one as the output.
    if (tensors.size() >= 1) {
      input = tensors[0];
      shape = input.shape();
    } else {
      std::cerr << "No tensors in file.\n";
      return 1;
    }
  }

  // If --file wasn't specified, set the shape from --shape, or randomly
  // if --seed was specified.
  if (vm.count("file") == 0) {
    if (vm.count("seed") != 0 && vm.count("shape") == 0) {
      std::cerr << "Randomly setting shape.\n";
      shape =
          getRandomShape(randomEngine, ipuModel.tilesPerIPU * ipuModel.numIPUs);
    } else {
      shape = parseSizeVector<std::size_t>(shapeString);
    }
  }
  if (vm.count("seed") == 0 && vm.count("initial-shape") != 0) {
    initialShape = parseSizeVector<std::size_t>(initialShapeString);
  }
  if (vm.count("seed") == 0 && vm.count("shuffle") != 0) {
    shuffle = parseSizeVector<unsigned>(shuffleString);
  }
  if (vm.count("seed") != 0 && vm.count("dims") == 0) {
    std::cerr << "Randomly setting dims.\n";
    dims = getRandomDims(randomEngine, shape.size());
  } else {
    dims = parseSizeVector<std::size_t>(dimsString);
  }

  if (vm.count("seed") != 0) {
    if (vm.count("operation") == 0) {
      std::cerr << "Randomly choosing reduction operation.\n";
      op = getRandomOp(randomEngine);
    }
    if (vm.count("scale") == 0) {
      std::cerr << "Randomly setting scale.\n";
      scale = getRandomScale(randomEngine, op);
    }
    if (vm.count("update") == 0) {
      std::cerr << "Randomly choosing + or +=.\n";
      update = getRandomUpdate(randomEngine, op);
    }
    if (vm.count("withoutput") == 0) {
      std::cerr << "Randomly choosing reduce() or reduceWithOutput().\n";
      withOutput = getRandomWithOutput(randomEngine);
    }
    if (vm.count("type") == 0) {
      std::cerr << "Choosing random data type.\n";
      dataType = getRandomTypes(randomEngine, op);
    }
    if (vm.count("computesetapi") == 0) {
      std::cerr << "Choosing random API.\n";
      computeSetApi = getRandomApi(randomEngine);
    }
  }

  // Verify the types.
  switch (op) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
  case popops::Operation::MUL:
  case popops::Operation::MIN:
  case popops::Operation::MAX:
    if (dataType == BOOL) {
      std::cerr << "Type cannot be bool for (SQUARE_)ADD, MUL, MIN or MAX.\n";
      return 1;
    }
    break;
  case popops::Operation::LOGICAL_AND:
  case popops::Operation::LOGICAL_OR:
    if (dataType != BOOL) {
      std::cerr << "Types must be bool for AND or OR.\n";
      return 1;
    }
    break;
  }

  // TODO: T12990 Some types of testing are not supported yet.

  // Boolean and int not supported because the reference implementation isn't
  // templated yet.
  if (op == popops::Operation::LOGICAL_AND ||
      op == popops::Operation::LOGICAL_OR) {
    std::cerr << "Testing currently doesn't support boolean ops. Setting to "
                 "float ADD.\n";
    op = popops::Operation::ADD;
    dataType = FLOAT;
  }
  if (dataType == INT) {
    std::cerr << "Int testing not supported yet. Setting to float.\n";
    dataType = FLOAT;
  }

  // Add the input if we didn't load it from a file.
  if (vm.count("file") == 0) {
    if (initialShape.size() && shuffle.size()) {
      std::cout
          << "Using the initial-shape and shuffle parameters to change the"
             " input tensor layout\n";
      input = graph.addVariable(dataType, initialShape, "input");
      mapTensorLinearly(graph, input);
      input = input.dimShuffle(shuffle);
      input = input.reshape(shape);
    } else {
      input = graph.addVariable(dataType, shape, "input");
      mapTensorLinearly(graph, input);
    }
  }

  if (op != popops::Operation::ADD && op != popops::Operation::SQUARE_ADD) {
    if (scale != 1.0f) {
      std::cerr << "Scale must be 1.0 for non-add operations.\n";
      scale = 1.0f;
    }
    if (update) {
      std::cerr << "Cannot use update for non-add operations. "
                   "Setting to false.\n";
      update = false;
    }
  }

  if (update && !withOutput) {
    std::cerr << "Update must use reduceWithOutput(). Using it.\n";
    withOutput = true;
  }

  // Output the settings.
  std::cerr << "Shape: { ";
  for (auto s : shape)
    std::cerr << s << " ";
  std::cerr << "}\n";
  std::cerr << "Dims: { ";
  for (auto s : dims)
    std::cerr << s << " ";
  std::cerr << "}\n";
  std::cerr << "Op: " << op << "\n";
  std::cerr << "Scale: " << scale << "\n";
  std::cerr << "Update: " << update << "\n";
  std::cerr << "WithOutput: " << withOutput << "\n";
  std::cerr << "NumIPUS: " << ipuModel.numIPUs << "\n";
  std::cerr << "TilesPerIPU: " << ipuModel.tilesPerIPU << "\n";
  std::cerr << "Type: " << dataType.toString() << "\n";
  std::cerr << "API: " << (computeSetApi ? "vector<ComputeSet>" : "Sequence")
            << "\n";

  std::cerr << "Generating reduction...\n";
  // Do the reduction
  Sequence prog;

  if (input.elementType() != dataType) {
    std::cerr << "Loaded tensor has type " << input.elementType().toString()
              << ", casting to " << dataType.toString() << "\n";
    input = popops::cast(graph, input, dataType, prog);
  }

  Tensor output;

  auto rate = graph.addConstant(FLOAT, {}, scale);
  graph.setTileMapping(rate, 0);
  const auto useScale =
      (op == popops::Operation::ADD || op == popops::Operation::SQUARE_ADD) &&
      scale != 1.0f;
  ReduceParams reductionParams;
  if (useScale) {
    reductionParams = {op, update, rate};
  } else {
    reductionParams = {op, update};
  }

  if (withOutput) {
    // Make the output.
    auto reducedShape = getReducedShape(input.shape(), dims);
    output = graph.addVariable(input.elementType(), reducedShape);
    mapTensorLinearly(graph, output);

    if (computeSetApi) {
      std::vector<ComputeSet> css;
      popops::reduceWithOutput(graph, input, output, dims, reductionParams,
                               css);
      for (const auto &cs : css) {
        prog.add(Execute(cs));
      }
    } else {
      popops::reduceWithOutput(graph, input, output, dims, reductionParams,
                               prog);
    }
  } else {
    if (computeSetApi) {
      std::vector<ComputeSet> css;
      output = popops::reduce(graph, input, dims, reductionParams, css);
      for (const auto &cs : css) {
        prog.add(Execute(cs));
      }
    } else {
      output = popops::reduce(graph, input, dims, reductionParams, prog);
    }
  }
  double absoluteTolerance = FLOAT_ABS_TOL;
  double relativeTolerance = FLOAT_REL_TOL;

  if (dataType == HALF) {
    absoluteTolerance = HALF_ABS_TOL;
    relativeTolerance = HALF_REL_TOL;
  }

  std::cerr << "Calculating reference...\n";

  poplibs_test::reduce::ReferenceTensor<double> inputTensor;
  inputTensor.shape = shape;
  inputTensor.values.resize(input.numElements());

  // Write random input values. Because we might have a lot of mul's, we want
  // the expected magnitude of the distribution to be 1 so that the final
  // expected magnitude is also 1.
  writeRandomValues(target, dataType, inputTensor.values.data(),
                    inputTensor.values.data() + inputTensor.values.size(), -2.0,
                    2.0, randomEngine);

  // Also write random values for the output for the `update` case.
  std::vector<double> outputValues(output.numElements());
  writeRandomValues(target, dataType, outputValues.data(),
                    outputValues.data() + outputValues.size(), -30.0, 30.0,
                    randomEngine);

  // Validate against a reference model
  auto outputRef = poplibs_test::reduce::reduce(inputTensor, dims, op);

  // Apply scale.
  for (auto &v : outputRef.values)
    v *= scale;

  // If it's an update, add the output values.
  if (update) {
    for (std::size_t i = 0; i < outputRef.values.size(); ++i)
      outputRef.values[i] += outputValues[i];
  }

  std::cerr << "Running engine...\n";

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto inputData = allocateHostMemoryForTensor(input, "input", graph,
                                               uploadProg, downloadProg, tmap);
  auto outputData = allocateHostMemoryForTensor(output, "output", graph,
                                                uploadProg, downloadProg, tmap);

  // Copy the input and output numbers to input/outputData, converting the
  // type as necessary.
  copy(target, inputTensor.values.data(), inputTensor.values.size(), dataType,
       inputData.get());
  copy(target, outputValues.data(), outputValues.size(), dataType,
       outputData.get());

  auto engineOptions = defaultEngineOptions;
  if (vm.count("profile")) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), engineOptions);
  attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  std::vector<double> outputTensor(output.numElements());

  copy(target, dataType, outputData.get(), outputTensor.data(),
       outputTensor.size());

  std::cerr << "Verifying result...\n";

  const bool matchesModel = checkIsClose(
      "reduce", outputTensor.data(), output.shape(), outputRef.values.data(),
      outputRef.values.size(), relativeTolerance, absoluteTolerance);
  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  std::cerr << "Validation succeeded!\n";
  return 0;
}
