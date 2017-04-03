#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <istream>
#include <ostream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popstd/ActivationMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popreduce/Reduce.hpp>
#include <popnn/Recurrent.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <poplib_test/GeneralMatrixMultiply.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <poplib_test/Rnn.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace popreduce;
using namespace poplin;
using namespace popstd;

namespace popnn {
std::istream &operator>>(std::istream &in, popnn::NonLinearityType &type) {
  std::string token;
  in >> token;
  if (token == "relu")
    type = popnn::NonLinearityType::NON_LINEARITY_RELU;
  else if (token == "sigmoid")
    type = popnn::NonLinearityType::NON_LINEARITY_SIGMOID;
  else if (token == "tanh")
    type = popnn::NonLinearityType::NON_LINEARITY_TANH;
  else
    throw poplib_test::poplib_test_error(
        "Unsupported nonlinearity <" + token + ">");

  return in;
}
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned sequenceSize, inputSize = 1, outputSize;
  unsigned batchSize = 1;

  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;

  popnn::NonLinearityType nonLinearityType =
                                popnn::NonLinearityType::NON_LINEARITY_SIGMOID;

  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("sequence-size", po::value<unsigned>(&sequenceSize)->required(),
     "Sequence size in the RNN")
    ("input-size", po::value<unsigned>(&inputSize)->default_value(inputSize),
     "Number of inputs in each element in the sequence. Must be specified if "
     "apply-feedforward-weights is set")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of outputs in each element in the sequence")
    ("nonlinearity-type",
     po::value<popnn::NonLinearityType>(&nonLinearityType)->
                                  default_value(nonLinearityType),
     "Non-linearity type: relu | sigmoid | tanh")
    ("apply-feedforward-weights",
     "Transform input by multipling it with input feedforward weights")
    ("data-type",
      po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
      "Input and output data type")
    ("batch-size", po::value<unsigned>(&batchSize)->default_value(batchSize),
      "Batch size")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
     "Type of the partials")
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&info.numIPUs)->default_value(info.numIPUs),
     "Number of IPUs")
  ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  bool applyFeedFwddWeights = vm.count("apply-feedforward-weights");

  if (applyFeedFwddWeights) {
    if (vm["input-size"].defaulted()) {
      std::cerr << "--input-size must be set if --apply-feedforward-weights "
      "is set\n";
      return 1;
    }
  }

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  Sequence prog;
  Tensor prevAct, feedFwdWeights, feedFwdOutput;

  if (applyFeedFwddWeights) {
    prevAct = graph.addTensor(dataTypeStr,
                              {sequenceSize, batchSize, inputSize}, "prevAct");
    for (unsigned s = 0U; s != sequenceSize; ++s) {
      mapActivations(graph, prevAct[s]);
    }
    PlanningCache cache;
    MatMulOptions mmOpt;
    mmOpt.partialsType = partialsTypeStr;
    mmOpt.leftHandArgUsedInTranspose = false;
    mmOpt.cache = &cache;
    feedFwdWeights = createMatMulInputA(graph, dataTypeStr,
                                        {outputSize, inputSize},
                                        prevAct[0].transpose(),
                                        "feedFwdWeights", mmOpt);

    feedFwdOutput = popnn::rnn::forwardWeightInput(graph, prevAct,
                                                   feedFwdWeights, prog,
                                                   partialsTypeStr, "");
  } else {
    feedFwdOutput = graph.addTensor(
                                   dataTypeStr,
                                   {sequenceSize, batchSize, outputSize},
                                   "feedFwdOutput");

    for (unsigned s = 0U; s != sequenceSize; ++s) {
      mapActivations(graph, feedFwdOutput[s]);
    }
  }

  Tensor initState = graph.addTensor(dataTypeStr,
                                     {batchSize, outputSize},
                                     "nextAct");
  /* This should be a single vector for all batches, but done so if in case
   * we want to split sequence steps over multiple calls
   */
  mapActivations(graph, initState);


  /* map biases and brooadcast them */
  auto biases = graph.addTensor(dataTypeStr, {outputSize}, "biases");
  mapTensor(graph, biases);

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = false;
  mmOpt.cache = &cache;


  auto feedbackWeights = createMatMulInputA(graph, dataTypeStr,
                                            {outputSize, outputSize},
                                            feedFwdOutput[0].transpose(),
                                            "feedbackWeights", mmOpt);

  auto nextAct = popnn::rnn::forwardIterate(graph, feedFwdOutput, initState,
                                            feedbackWeights,
                                            biases, prog, nonLinearityType,
                                            partialsTypeStr, "");

  auto upload = Sequence();
  auto download = Sequence();

  std::unique_ptr<char[]> rawHostPrevAct;
  std::unique_ptr<char[]> rawHostFeedFwdWeights;
  std::vector< std::unique_ptr<char[]> > rawHostfeedFwdOutput;
  std::vector< std::unique_ptr<char[]> > rawHostNextAct;


  if (applyFeedFwddWeights) {
    rawHostPrevAct = allocateHostMemoryForTensor(
                        graph, prevAct, upload, download);
    rawHostFeedFwdWeights = allocateHostMemoryForTensor(
                        graph, feedFwdWeights, upload, download);
  }

  for (auto s = 0U; s != sequenceSize; ++s) {
    rawHostfeedFwdOutput.push_back(allocateHostMemoryForTensor(
                                    graph, feedFwdOutput[s], upload, download));
    rawHostNextAct.push_back(allocateHostMemoryForTensor(
                                    graph, nextAct[s], upload, download));

  }

  auto rawHostFeedbackWeights = allocateHostMemoryForTensor(
                        graph, feedbackWeights, upload, download);
  auto rawHostInitState = allocateHostMemoryForTensor(
                        graph, initState, upload, download);
  auto rawHostBiases = allocateHostMemoryForTensor(
                        graph, biases, upload, download);

  Engine engine(graph, {std::move(upload),
                        std::move(download),
                        std::move(prog)});

  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 2>
      hostFeedFwdWeights(boost::extents[outputSize][inputSize]);
  boost::multi_array<double, 2>
      hostFeedbackWeights(boost::extents[outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostfeedFwdOutput(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      modelfeedFwdOutput(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[outputSize]);
  boost::multi_array<double, 2>
      hostInitState(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 3>
      modelNextAct(boost::extents[sequenceSize][batchSize][outputSize]);

  std::fill(hostInitState.data(),
            hostInitState.data() + hostInitState.num_elements(),
            0);

  std::mt19937 randomEngine;

  if (applyFeedFwddWeights) {
    writeRandomValues(hostPrevAct, -4.0, 4.0, randomEngine);
    writeRandomValues(hostFeedFwdWeights, -3.0, 3.0, randomEngine);
    poplib_test::rnn::forwardWeightInput(hostPrevAct, hostFeedFwdWeights,
                                         modelfeedFwdOutput);
  }

  writeRandomValues(hostFeedbackWeights, -2.0, 2.0, randomEngine);
  writeRandomValues(hostBiases, -1.0, 1.0, randomEngine);

  poplib_test::rnn::forwardIterate(applyFeedFwddWeights
                                          ? modelfeedFwdOutput :
                                            hostfeedFwdOutput,
                                   hostInitState,
                                   hostFeedbackWeights, hostBiases,
                                   modelNextAct, nonLinearityType);

  if (applyFeedFwddWeights) {
    copy(hostPrevAct, dataTypeStr, rawHostPrevAct.get());
    copy(hostFeedFwdWeights, dataTypeStr, rawHostFeedFwdWeights.get());
  } else {
    for (auto s = 0U; s != rawHostfeedFwdOutput.size(); ++s) {
      boost::multi_array<double, 2> subMat = hostfeedFwdOutput[s];
      copy(subMat, dataTypeStr, rawHostfeedFwdOutput[s].get());
    }
  }

  copy(hostFeedbackWeights, dataTypeStr, rawHostFeedbackWeights.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());
  copy(hostInitState, dataTypeStr, rawHostInitState.get());

  engine.run(0);    // Upload
  engine.run(2);    // matrix operation
  engine.run(1);    // download

  bool matchesModel = false;

  if (applyFeedFwddWeights) {
    for (auto s = 0U; s != rawHostfeedFwdOutput.size(); ++s) {
      boost::multi_array<double, 2>
          impSubMat(boost::extents[batchSize][outputSize]);
      copy(dataTypeStr, rawHostfeedFwdOutput[s].get(), impSubMat);
      boost::multi_array<double, 2> refSubMat = modelfeedFwdOutput[s];
      matchesModel |= checkIsClose("feedFwdOutput", impSubMat, refSubMat,
                                   relativeTolerance);
    }
  }

  for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
    boost::multi_array<double, 2>
        impSubMat(boost::extents[batchSize][outputSize]);
    copy(dataTypeStr, rawHostNextAct[s].get(), impSubMat);
    boost::multi_array<double, 2> refSubMat = modelNextAct[s];
    matchesModel |= checkIsClose("nextAct", impSubMat, refSubMat,
                                  relativeTolerance);
  }

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
