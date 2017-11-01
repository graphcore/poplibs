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
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/Lstm.hpp>
#include <popstd/TileMapping.hpp>
#include <poplar/HalfFloat.hpp>
#include <popconv/codelets.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <popstd/Zero.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poplib_test/Lstm.hpp>
#include <poplib_test/Util.hpp>
#include <poplib_test/Pass.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace poplin;
using namespace popstd;
using namespace popnn;

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;

  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  double absoluteTolerance;

  IPUModel ipuModel;
  ipuModel.IPUExchangeType =
      IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;
  bool preweightInput = false;
  poplib_test::Pass pass = poplib_test::Pass::FWD;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("sequence-size", po::value<unsigned>(&sequenceSize)->required(),
     "Sequence size in the RNN")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs in each element in the sequence")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of outputs in each element in the sequence")
    ("data-type",
      po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
      "Input and output data type")
    ("batch-size", po::value<unsigned>(&batchSize)->default_value(batchSize),
      "Batch size")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
     "Type of the partials")
    ("rel-tolerance", po::value<double>(&relativeTolerance)->
     default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("abs-tolerance",po::value<double>(&absoluteTolerance)->default_value(1e-6),
     "Absolute tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->default_value(ipuModel.numIPUs),
     "Number of IPUs")
    ("pre-weight-input",
       po::value<bool>(&preweightInput)->default_value(preweightInput),
     "Pre-weight whole sequence before recursive part is computed (0 / 1)")
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
  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  auto device = ipuModel.createDevice();
  Graph graph(device);
  popconv::addCodelets(graph);
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  // Bwd pass is always run if WU is run. This may change is tensors input to
  //  WU are created on host
  bool doBwdPass = pass == poplib_test::Pass::ALL
                   || pass == poplib_test::Pass::BWD
                   || pass == poplib_test::Pass::WU;
  bool doWuPass = pass == poplib_test::Pass::ALL
                  || pass == poplib_test::Pass::WU;
  bool fwdOnly = !doBwdPass && !doWuPass;

  auto prevLayerAct =
    lstm::createInput(graph, sequenceSize, batchSize, inputSize, outputSize,
                      dataTypeStr, fwdOnly, "prevLayerAct");
  auto prog = Sequence();
  auto fwdStateInit =
    lstm::createFwdState(graph, batchSize, outputSize, prog, false, dataTypeStr,
                         false);

  auto outputInit = lstm::getOutputFromFwdState(fwdStateInit);
  auto cellStateInit = lstm::getCellFromFwdState(fwdStateInit);

  auto biases = graph.addTensor(dataTypeStr,
                                {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                "biases");
  auto weightsInput =
    lstm::createWeightsInput(graph, sequenceSize, batchSize, inputSize,
                             outputSize, false, dataTypeStr, partialsTypeStr,
                             fwdOnly);
  auto weightsOutput =
    lstm::createWeightsOutput(graph, sequenceSize, batchSize, outputSize,
                              dataTypeStr, partialsTypeStr, fwdOnly);

  std::vector<std::pair<std::string, char *>> tmap;

  Tensor fwdState;
  if (preweightInput) {
    auto weightedIn =
      lstm::calcSequenceWeightedInputs(graph, prevLayerAct, weightsInput, prog,
                                       partialsTypeStr);
    auto prevOutAct = popnn::lstm::getOutputFromFwdState(fwdStateInit);
    auto prevCellState = popnn::lstm::getCellFromFwdState(fwdStateInit);
    fwdState =
      lstm::basicLstmCellForwardPassWeightedInputs(graph, weightedIn, biases,
                                                   prevOutAct, prevCellState,
                                                   weightsOutput,
                                                   prog, partialsTypeStr,
                                                   fwdOnly);
  } else {
    auto prevOutAct = popnn::lstm::getOutputFromFwdState(fwdStateInit);
    auto prevCellState = popnn::lstm::getCellFromFwdState(fwdStateInit);
    fwdState =
      lstm::basicLstmCellForwardPass(graph, prevLayerAct, biases, prevOutAct,
                                     prevCellState, weightsInput, weightsOutput,
                                     prog, partialsTypeStr, fwdOnly);
  }

  auto nextLayerGrads =
     graph.addTensor(dataTypeStr, {sequenceSize, batchSize, outputSize});
  mapTensorLinearly(graph, nextLayerGrads);

  Tensor bwdStateInit;
  if (doBwdPass || doWuPass) {
    bwdStateInit =
      lstm::createBwdState(graph, batchSize, outputSize, prog, dataTypeStr);
  }

  Tensor bwdState;
  Tensor weightsInputDeltas, weightsOutputDeltas, biasDeltas;

  if (doWuPass) {
    weightsInputDeltas = graph.clone(weightsInput);
    weightsOutputDeltas = graph.clone(weightsOutput);
    biasDeltas = graph.clone(biases);
    popstd::zero(graph, weightsInputDeltas, prog);
    popstd::zero(graph, weightsOutputDeltas, prog);
    popstd::zero(graph, biasDeltas, prog);
  }

  Tensor prevLayerGrads;
  if (doBwdPass || doWuPass) {
    std::vector<Tensor> prevLayerGradsVec(sequenceSize);
    for (auto i = sequenceSize; i != 0; --i) {
      const auto s = i - 1;
      auto bwdStateThisStep = s == sequenceSize - 1 ? bwdStateInit : bwdState;
      auto fwdStatePrevStep = s == 0 ? fwdStateInit : fwdState[s - 1];
      auto prevCellState =
        lstm::getCellFromFwdState(s == 0 ? fwdStateInit : fwdState[s - 1]);
      Tensor gradIn;
      std::tie(gradIn, bwdState) =
        lstm::basicLstmBackwardStep(graph, nextLayerGrads[s], fwdState[s],
                                    prevCellState, bwdStateThisStep,
                                    weightsInput, weightsOutput, prog,
                                    partialsTypeStr);
      prevLayerGradsVec[s] = gradIn.expand({0});

      if (doWuPass) {
        auto outState = s == 0 ? fwdStateInit : fwdState[s - 1];
        lstm::basicLstmParamUpdate(graph, prevLayerAct[s],
                                   lstm::getOutputFromFwdState(outState),
                                   bwdState, weightsInputDeltas,
                                   weightsOutputDeltas, biasDeltas, prog,
                                   partialsTypeStr);
      }
    }
    prevLayerGrads = concat(prevLayerGradsVec);
  }

  auto rawHostWeightsInput =
    allocateHostMemoryForTensor(weightsInput, "weightsInput", graph, tmap);
  auto rawHostWeightsOutput =
    allocateHostMemoryForTensor(weightsOutput, "weightsOutput", graph, tmap);
  auto rawHostPrevLayerAct =
    allocateHostMemoryForTensor(prevLayerAct, "prevLayerAct", graph, tmap);
  auto rawHostBiases =
    allocateHostMemoryForTensor(biases, "biases", graph, tmap);
  auto rawHostOutputInit =
    allocateHostMemoryForTensor(outputInit, "outputInit", graph, tmap);
  auto rawHostCellStateInit =
    allocateHostMemoryForTensor(cellStateInit, "cellStateInit", graph, tmap);

  std::unique_ptr<char[]> rawHostNextLayerGrads;
  std::unique_ptr<char[]> rawHostPrevLayerGrads;
  std::unique_ptr<char[]> rawHostWeightsInputDeltas;
  std::unique_ptr<char[]> rawHostWeightsOutputDeltas;
  std::unique_ptr<char[]> rawHostBiasDeltas;

  if (doBwdPass) {
    rawHostNextLayerGrads =
      allocateHostMemoryForTensor(nextLayerGrads, "nextLayerGrads", graph,
                                  tmap);
    rawHostPrevLayerGrads =
      allocateHostMemoryForTensor(prevLayerGrads, "prevLayerGrads", graph,
                                  tmap);
  }
  if (doWuPass) {
    rawHostWeightsInputDeltas =
      allocateHostMemoryForTensor(weightsInputDeltas, "weightsInputDeltas",
                                  graph, tmap);
    rawHostWeightsOutputDeltas =
      allocateHostMemoryForTensor(weightsOutputDeltas, "weightsOutputDeltas",
                                  graph, tmap);
    rawHostBiasDeltas =
      allocateHostMemoryForTensor(biasDeltas, "biasDeltas", graph, tmap);
  }

  std::vector<std::unique_ptr<char[]>> rawHostNextAct;
  for (auto s = 0U; s != sequenceSize; ++s) {
    auto nextAct = lstm::getOutputFromFwdState(fwdState[s]);
    rawHostNextAct.push_back(allocateHostMemoryForTensor(nextAct,
                                                         "nextAct" +
                                                           std::to_string(s),
                                                         graph, tmap));
  }

  Engine engine(device, graph, prog);

  boost::multi_array<double, 3>
      hostPrevLayerAct(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      hostWeightsOutput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsInput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostBiases(boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2>
      hostCellStateInit(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      modelCellState(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      hostOutputInit(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 4>
      modelFwdState(boost::extents[LSTM_NUM_FWD_STATES][sequenceSize]
                                  [batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostNextLayerGrads(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      modelPrevLayerGrads(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 4>
      modelBwdState(boost::extents[LSTM_NUM_BWD_STATES][sequenceSize]
                                  [batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsOutputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                            [outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsInputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                           [inputSize][outputSize]);
  boost::multi_array<double, 2>
      hostBiasesDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);

  std::mt19937 randomEngine;

  writeRandomValues(hostPrevLayerAct, -4.0, 4.0, randomEngine);
  writeRandomValues(hostOutputInit, -3.0, 3.0, randomEngine);
  writeRandomValues(hostCellStateInit, -3.0, 3.0, randomEngine);
  writeRandomValues(hostWeightsInput, -1.0, 1.0, randomEngine);
  writeRandomValues(hostWeightsOutput, -1.0, 1.0, randomEngine);
  writeRandomValues(hostBiases, -1.0, 1.0, randomEngine);

  if (doBwdPass) {
    writeRandomValues(hostNextLayerGrads, -2.0, 2.0, randomEngine);
  }

  modelCellState = hostCellStateInit;

  copy(hostPrevLayerAct, dataTypeStr, rawHostPrevLayerAct.get());
  copy(hostCellStateInit, dataTypeStr, rawHostCellStateInit.get());
  copy(hostOutputInit, dataTypeStr, rawHostOutputInit.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());
  copy(hostWeightsInput, dataTypeStr, rawHostWeightsInput.get());
  copy(hostWeightsOutput, dataTypeStr, rawHostWeightsOutput.get());
  if (doBwdPass) {
    copy(hostNextLayerGrads, dataTypeStr, rawHostNextLayerGrads.get());
  }

  upload(engine, tmap);
  engine.run(0);
  download(engine, tmap);

  poplib_test::lstm::basicLstmCellForwardPass(
                          hostPrevLayerAct, hostBiases, hostOutputInit,
                          hostWeightsInput, hostWeightsOutput, modelCellState,
                          modelFwdState);

  if (doBwdPass) {
    poplib_test::lstm::basicLstmCellBackwardPass(
                            hostWeightsInput, hostWeightsOutput,
                            hostNextLayerGrads, hostCellStateInit,
                            modelFwdState,  modelBwdState, modelPrevLayerGrads);
  }

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);
  bool matchesModel = true;

  for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
    boost::multi_array<double, 2> subMatImp(boost::extents[batchSize]
                                                          [outputSize]);
    copy(dataTypeStr, rawHostNextAct[s].get(), subMatImp);
    boost::multi_array<double, 2> subMatRef =
        modelFwdState[LSTM_FWD_STATE_ACTS_IDX][s];
    matchesModel &= checkIsClose("nextLayerAct", subMatRef, subMatImp,
                                 relativeTolerance);
  }

  if (doBwdPass) {
    copy(dataTypeStr, rawHostPrevLayerGrads.get(), hostPrevLayerGrads);

    matchesModel &=
      checkIsClose("prevLayerGrads", modelPrevLayerGrads, hostPrevLayerGrads,
                   relativeTolerance, absoluteTolerance);
  }

  if (doWuPass) {
    copy(dataTypeStr, rawHostWeightsInputDeltas.get(), hostWeightsInputDeltas);
    copy(dataTypeStr, rawHostWeightsOutputDeltas.get(),
         hostWeightsOutputDeltas);
    copy(dataTypeStr, rawHostBiasDeltas.get(), hostBiasesDeltas);
    boost::multi_array<double, 3>
        modelWeightsOutputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                              [outputSize][outputSize]);
    boost::multi_array<double, 3>
        modelWeightsInputDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                             [inputSize][outputSize]);
    boost::multi_array<double, 2>
        modelBiasesDeltas(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                        [outputSize]);
    poplib_test::lstm::basicLstmCellParamUpdate(
                            hostPrevLayerAct, modelFwdState, hostOutputInit,
                            modelBwdState, modelWeightsInputDeltas,
                            modelWeightsOutputDeltas, modelBiasesDeltas);
    matchesModel &=
        checkIsClose("weightsInputDeltas", modelWeightsInputDeltas,
                     hostWeightsInputDeltas,relativeTolerance,
                     absoluteTolerance);
    matchesModel &=
      checkIsClose("weightsOutputDeltas", modelWeightsOutputDeltas,
                   hostWeightsOutputDeltas, relativeTolerance,
                   absoluteTolerance);
    matchesModel &=
        checkIsClose("biasDeltas", modelBiasesDeltas, hostBiasesDeltas,
                     relativeTolerance, absoluteTolerance);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
