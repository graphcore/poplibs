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
#include <popconv/codelets.hpp>
#include <popstd/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/Zero.hpp>
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
#include <poplib_test/Pass.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace popreduce;
using namespace poplin;
using namespace popstd;


const char *asString(popnn::NonLinearityType &type) {
  switch (type) {
  case popnn::NonLinearityType::NON_LINEARITY_RELU: return "relu";
  case popnn::NonLinearityType::NON_LINEARITY_SIGMOID: return "sigmoid";
  case popnn::NonLinearityType::NON_LINEARITY_TANH: return "tanh";
  case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX: return "softmax";
  }
  POPLIB_UNREACHABLE();
}

std::ostream &operator<<(std::ostream &os, popnn::NonLinearityType &type) {
  return os << asString(type);
}

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
  double relativeTolerance, absoluteTolerance;

  popnn::NonLinearityType nonLinearityType =
                                popnn::NonLinearityType::NON_LINEARITY_SIGMOID;

  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;

  poplib_test::Pass pass = poplib_test::Pass::FWD;
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
    ("rel-tolerance",po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("abs-tolerance",po::value<double>(&absoluteTolerance)->default_value(1e-6),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&info.numIPUs)->default_value(info.numIPUs),
     "Number of IPUs")
    ("phase",
     po::value<poplib_test::Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
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

  bool applyFeedFwdWeights = vm.count("apply-feedforward-weights");

  if (applyFeedFwdWeights) {
    if (vm["input-size"].defaulted()) {
      std::cerr << "--input-size must be set if --apply-feedforward-weights "
      "is set\n";
      return 1;
    }
  }

  bool doBwdPass = pass == poplib_test::Pass::ALL
                   || pass == poplib_test::Pass::BWD;
  bool doWuPass = pass == poplib_test::Pass::ALL
                  || pass == poplib_test::Pass::WU;
  bool fwdOnly = !doBwdPass && !doWuPass;

  // force appication of feed-forward weights if bwd pass or fwd pass is enabled
  if ((doBwdPass || doWuPass) && !applyFeedFwdWeights) {
    applyFeedFwdWeights = true;
  }

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  Graph graph(createIPUModelDevice(info));
  popconv::addCodelets(graph);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  Sequence prog;
  Tensor prevAct, feedFwdWeights, feedFwdOutput;

  if (applyFeedFwdWeights) {
    prevAct = popnn::rnn::createInput(graph, sequenceSize, batchSize, inputSize,
                                      outputSize, dataTypeStr, partialsTypeStr,
                                      fwdOnly);
    feedFwdWeights =
        popnn::rnn::createWeightsInput(graph, sequenceSize, batchSize,
                                       inputSize, outputSize,
                                       dataTypeStr, partialsTypeStr, fwdOnly);
    feedFwdOutput = popnn::rnn::forwardWeightInput(graph, prevAct,
                                                   feedFwdWeights, prog,
                                                   partialsTypeStr, "");
  } else {
    feedFwdOutput = graph.addTensor(dataTypeStr,
                                         {0, batchSize, outputSize},
                                         "feedFwdOutput");
    for (unsigned s = 0U; s != sequenceSize; ++s) {
      auto h =
        popnn::rnn::createFwdState(graph, dataTypeStr, batchSize, outputSize,
                                   prog, false);

      feedFwdOutput = append(feedFwdOutput,
                             popnn::rnn::getOutputFromFwdState(h));
    }
  }

  auto fwdInitState =
    popnn::rnn::createFwdState(graph, dataTypeStr, batchSize, outputSize, prog,
                               false);
  auto initAct =  popnn::rnn::getOutputFromFwdState(fwdInitState);

  /* map biases and brooadcast them */
  auto biases = graph.addTensor(dataTypeStr, {outputSize}, "biases");
  mapTensorLinearly(graph, biases);

  auto feedbackWeights =
    popnn::rnn::createWeightsFeedback(graph, batchSize,outputSize,
                                        dataTypeStr, partialsTypeStr, fwdOnly);

  auto fwdNextState =
    popnn::rnn::forwardIterate(graph, feedFwdOutput, fwdInitState,
                               feedbackWeights, biases, prog, nonLinearityType,
                               partialsTypeStr, "");

  Tensor nextLayerGrads;
  if (doBwdPass || doWuPass) {
    nextLayerGrads = graph.addTensor(dataTypeStr,
                                     {sequenceSize, batchSize, outputSize},
                                     "nextLayerGrads");
    mapTensorLinearly(graph, nextLayerGrads);
  }

  Tensor feedFwdWeightsDeltaAcc, feedbackWeightsDeltaAcc, biasesDeltaAcc;
  Tensor prevLayerGradsThisStep, bwdState;

  if (doBwdPass || doWuPass) {
    bwdState = popnn::rnn::createBwdState(graph, dataTypeStr, batchSize,
                                               outputSize, prog);
  }

  if (doWuPass) {
    feedFwdWeightsDeltaAcc = graph.clone(feedFwdWeights);
      feedbackWeightsDeltaAcc = graph.clone(feedbackWeights);
      biasesDeltaAcc = graph.clone(biases);
      // zero all tensors updated in the BPTT
      zero(graph, feedFwdWeightsDeltaAcc, prog, "ZeroFeedFwdWeightsDeltasAcc");
      zero(graph, feedbackWeightsDeltaAcc, prog,
           "ZeroFeedbackWeightsDeltasAcc");
      zero(graph, biasesDeltaAcc, prog, "ZeroBiasesDeltasAcc");
  }

  std::vector<Tensor> prevLayerGradsVec(sequenceSize);
  std::vector<Tensor> gradientSumVec(sequenceSize);

  for (auto i = sequenceSize; i != 0; --i) {
    auto s = i - 1;
    if (doBwdPass || doWuPass) {
      std::tie(prevLayerGradsThisStep, bwdState) =
        popnn::rnn::backwardGradientStep(graph, nextLayerGrads[s],
                                         bwdState,
                                         fwdNextState[s], feedFwdWeights,
                                         feedbackWeights, prog,
                                         nonLinearityType);
      gradientSumVec[s] = bwdState.expand({0});
      prevLayerGradsVec[s] = prevLayerGradsThisStep.expand({0});
    }
    if (doWuPass) {
      Tensor state = s == 0 ? fwdInitState : fwdNextState[s - 1];
      popnn::rnn::paramDeltaUpdate(graph, bwdState, prevAct[s],
                                   state, feedFwdWeightsDeltaAcc,
                                   feedbackWeightsDeltaAcc, biasesDeltaAcc,
                                   prog);
    }
  }

  Tensor prevLayerGrads, gradientSum;
  if (doBwdPass || doWuPass) {
    prevLayerGrads = concat(prevLayerGradsVec);
    gradientSum = concat(gradientSumVec);
  }
  std::unique_ptr<char[]> rawHostPrevAct;
  std::unique_ptr<char[]> rawHostFeedFwdWeights;
  std::vector< std::unique_ptr<char[]> > rawHostfeedFwdOutput;
  std::vector< std::unique_ptr<char[]> > rawHostNextAct;
  std::vector<std::pair<std::string, char *>> tmap;
  if (applyFeedFwdWeights) {
    rawHostPrevAct = allocateHostMemoryForTensor(prevAct, "prevAct", graph,
                                                 tmap);
    rawHostFeedFwdWeights = allocateHostMemoryForTensor(feedFwdWeights,
                                                        "feedFwdWeights",
                                                        graph, tmap);
  }

  for (auto s = 0U; s != sequenceSize; ++s) {
    rawHostfeedFwdOutput.push_back(
         allocateHostMemoryForTensor(feedFwdOutput[s],
                                     "feedFwdOutput" + std::to_string(s),
                                     graph, tmap));
    auto nextAct = popnn::rnn::getOutputFromFwdState(fwdNextState[s]);
    rawHostNextAct.push_back(
        allocateHostMemoryForTensor(nextAct, "nextAct" + std::to_string(s),
                                    graph, tmap));
  }

  auto rawHostFeedbackWeights =
      allocateHostMemoryForTensor(feedbackWeights, "feedbackWeights",
                                  graph, tmap);
  auto rawHostInitAct =
      allocateHostMemoryForTensor(initAct, "initAct", graph, tmap);
  auto rawHostBiases =
      allocateHostMemoryForTensor(biases, "biases", graph, tmap);

  std::unique_ptr<char[]> rawNextLayerGrads;
  std::unique_ptr<char[]> rawHostPrevLayerGrads;
  std::unique_ptr<char[]> rawHostGradientSum;
  if (doBwdPass || doWuPass) {
    rawNextLayerGrads =
      allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                  tmap);
    rawHostPrevLayerGrads =
      allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                  tmap);
    rawHostGradientSum =
      allocateHostMemoryForTensor(gradientSum, "gradientSum", graph, tmap);
  }
  std::unique_ptr<char[]> rawHostFeedFwdWeightsDeltasAcc;
  std::unique_ptr<char[]> rawHostFeedbackWeightsDeltasAcc;
  std::unique_ptr<char[]> rawHostBiasesDeltasAcc;
  if (doWuPass) {
    rawHostFeedFwdWeightsDeltasAcc =
        allocateHostMemoryForTensor(feedFwdWeightsDeltaAcc,
                                    "feedFwdWeightsDeltaAcc", graph, tmap);
    rawHostFeedbackWeightsDeltasAcc =
        allocateHostMemoryForTensor(feedbackWeightsDeltaAcc,
                                    "feedbackWeightsDeltaAcc", graph, tmap);
    rawHostBiasesDeltasAcc =
        allocateHostMemoryForTensor(biasesDeltaAcc,
                                    "biasesDeltaAcc", graph, tmap);
  }

  Engine engine(graph, prog);

  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 2>
      hostFeedFwdWeights(boost::extents[inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostFeedbackWeights(boost::extents[outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostfeedFwdOutput(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      modelfeedFwdOutput(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 1>
      hostBiases(boost::extents[outputSize]);
  boost::multi_array<double, 2>
      hostInitAct(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 3>
      modelNextAct(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostNextLayerGrads(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      hostGradientSum(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 2>
      hostFeedFwdWeightsDeltasAcc(boost::extents[inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostFeedbackWeightsDeltasAcc(boost::extents[outputSize][outputSize]);
  boost::multi_array<double, 1>
      hostBiasesDeltasAcc(boost::extents[outputSize]);

  std::fill(hostInitAct.data(),
            hostInitAct.data() + hostInitAct.num_elements(),
            0);

  std::mt19937 randomEngine;

  if (applyFeedFwdWeights) {
    writeRandomValues(hostPrevAct, -4.0, 4.0, randomEngine);
    writeRandomValues(hostFeedFwdWeights, -3.0, 3.0, randomEngine);
    poplib_test::rnn::forwardWeightInput(hostPrevAct, hostFeedFwdWeights,
                                         modelfeedFwdOutput);
  }

  writeRandomValues(hostFeedbackWeights, -2.0, 2.0, randomEngine);
  writeRandomValues(hostBiases, -1.0, 1.0, randomEngine);
  writeRandomValues(hostNextLayerGrads, -1.0, 1.0, randomEngine);

  poplib_test::rnn::forwardIterate(applyFeedFwdWeights
                                          ? modelfeedFwdOutput :
                                            hostfeedFwdOutput,
                                   hostInitAct,
                                   hostFeedbackWeights, hostBiases,
                                   modelNextAct, nonLinearityType);

  boost::multi_array<double, 3>
      modelPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      modelGradientSum(boost::extents[sequenceSize][batchSize][outputSize]);

  if (doBwdPass || doWuPass) {
    poplib_test::rnn::backward(modelNextAct, hostNextLayerGrads,
                               hostFeedFwdWeights, hostFeedbackWeights,
                               modelPrevLayerGrads, modelGradientSum,
                               nonLinearityType);
  }

  boost::multi_array<double, 2>
      modelFeedFwdWeightsDeltasAcc(boost::extents[inputSize][outputSize]);
  boost::multi_array<double, 2>
      modelFeedbackWeightsDeltasAcc(boost::extents[outputSize][outputSize]);
  boost::multi_array<double, 1>
      modelBiasesDeltasAcc(boost::extents[outputSize]);
  if (doWuPass) {
    poplib_test::rnn::paramUpdate(hostPrevAct, hostInitAct, modelNextAct,
                                  modelGradientSum,
                                  modelFeedFwdWeightsDeltasAcc,
                                  modelFeedbackWeightsDeltasAcc,
                                  modelBiasesDeltasAcc);
  }

  if (applyFeedFwdWeights) {
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
  copy(hostInitAct, dataTypeStr, rawHostInitAct.get());

  if (doBwdPass || doWuPass) {
    copy(hostNextLayerGrads, dataTypeStr, rawNextLayerGrads.get());
  }

  upload(engine, tmap);
  engine.run(0);
  download(engine, tmap);

  bool matchesModel = true;

  if (applyFeedFwdWeights) {
    for (auto s = 0U; s != rawHostfeedFwdOutput.size(); ++s) {
      boost::multi_array<double, 2>
          impSubMat(boost::extents[batchSize][outputSize]);
      copy(dataTypeStr, rawHostfeedFwdOutput[s].get(), impSubMat);
      boost::multi_array<double, 2> refSubMat = modelfeedFwdOutput[s];
      matchesModel &= checkIsClose("feedFwdOutput", impSubMat, refSubMat,
                                   relativeTolerance, absoluteTolerance);
    }
  }

  for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
    boost::multi_array<double, 2>
        impSubMat(boost::extents[batchSize][outputSize]);
    copy(dataTypeStr, rawHostNextAct[s].get(), impSubMat);
    boost::multi_array<double, 2> refSubMat = modelNextAct[s];
    matchesModel &= checkIsClose("nextAct", impSubMat, refSubMat,
                                  relativeTolerance, absoluteTolerance);
  }

  if (doWuPass || doBwdPass) {
    copy(dataTypeStr, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);
    copy(dataTypeStr, rawHostGradientSum.get(), hostGradientSum);
  }
  if (doWuPass) {
    copy(dataTypeStr, rawHostFeedFwdWeightsDeltasAcc.get(),
         hostFeedFwdWeightsDeltasAcc);
    copy(dataTypeStr, rawHostFeedbackWeightsDeltasAcc.get(),
         hostFeedbackWeightsDeltasAcc);
    copy(dataTypeStr, rawHostBiasesDeltasAcc.get(), hostBiasesDeltasAcc);
  }

  if (doBwdPass) {
    for (auto s = 0; s != sequenceSize; ++s) {
      auto seqStr = std::to_string(s);
      boost::multi_array<double, 2> gradInputRef = modelPrevLayerGrads[s];
      boost::multi_array<double, 2> gradInputImpl = hostPrevLayerGrads[s];
      matchesModel &=
          checkIsClose("prevLayerGrad/" + seqStr, gradInputImpl, gradInputRef,
                       relativeTolerance, absoluteTolerance);
      boost::multi_array<double, 2> gradSumRef = modelGradientSum[s];
      boost::multi_array<double, 2> gradSumImpl = hostGradientSum[s];
      matchesModel &=
          checkIsClose("gradientSum/" + seqStr, gradSumImpl, gradSumRef,
                       relativeTolerance, absoluteTolerance);
    }
  }

  if (doWuPass) {
    matchesModel &=
      checkIsClose("FeedFwdWeightsDeltasAcc", hostFeedFwdWeightsDeltasAcc,
                   modelFeedFwdWeightsDeltasAcc, relativeTolerance,
                   absoluteTolerance);
    matchesModel &=
      checkIsClose("FeedbackWeightsDeltasAcc", hostFeedbackWeightsDeltasAcc,
                   modelFeedbackWeightsDeltasAcc, relativeTolerance,
                   absoluteTolerance);
    matchesModel &=
      checkIsClose("BiasesDeltasAcc", hostBiasesDeltasAcc,
                   modelBiasesDeltasAcc, relativeTolerance, absoluteTolerance);
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
