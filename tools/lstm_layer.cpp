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
#include <poplin/MatMul.hpp>
#include <popnn/Lstm.hpp>
#include <popstd/TileMapping.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poplib_test/Lstm.hpp>
#include <poplib_test/Util.hpp>
#include <util/Compiler.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplib_test::util;
using namespace poplin;
using namespace popstd;


int main(int argc, char **argv) {
  namespace po = boost::program_options;

  unsigned sequenceSize, inputSize, outputSize;
  unsigned batchSize = 1;

  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;

  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;

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

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  Graph graph(createIPUModelDevice(info));
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);

  Tensor prevAct = graph.addTensor(dataTypeStr,
                                   {sequenceSize, batchSize, inputSize},
                                   "prevAct");
  for (unsigned s = 0U; s != sequenceSize; ++s) {
    mapTensorLinearly(graph, prevAct[s]);
  }

  /* This should be a single vector for all batches, but done so if in case
   * we want to split sequence steps over multiple calls
   */
  Tensor prevOutput = graph.addTensor(dataTypeStr,
                                      {batchSize, outputSize},
                                      "prevOutput");
  mapTensorLinearly(graph, prevOutput);

  Tensor cellState = graph.addTensor(dataTypeStr,
                                     {batchSize, outputSize},
                                     "cellState");
  mapTensorLinearly(graph, cellState);

  /* map biases and brooadcast them */
  auto biases = graph.addTensor(dataTypeStr,
                                {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                "biases");
  auto prog = Sequence();

  /* map weights */
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = false;
  mmOpt.cache = &cache;

  Tensor weightsOutput = graph.addTensor(dataTypeStr,
                                         {0, outputSize, outputSize},
                                         "");
  Tensor weightsInput = graph.addTensor(dataTypeStr,
                                         {0, outputSize, inputSize},
                                         "");

  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<std::unique_ptr<char[]>> rawHostWeightsInput;
  std::vector<std::unique_ptr<char[]>> rawHostWeightsOutput;
  for (auto u = 0U; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto wName = "weightsOutput" + std::to_string(u);
    auto wOut = createMatMulInputA(graph, dataTypeStr,
                                   {outputSize, outputSize},
                                   cellState.transpose(),
                                   wName, mmOpt);

    weightsOutput = append(weightsOutput, wOut);

    rawHostWeightsOutput.push_back(allocateHostMemoryForTensor(wOut,
                                                               wName + "out",
                                                               graph, tmap));

    auto wInp =
        createMatMulInputA(graph, dataTypeStr, {outputSize, inputSize},
                           prevAct[0].transpose(),
                           "weightsInput" + std::to_string(u), mmOpt);
    weightsInput = append(weightsInput, wInp);
    rawHostWeightsInput.push_back(allocateHostMemoryForTensor(wInp,
                                                              wName + "in",
                                                              graph, tmap));
  }

  Tensor nextAct =
      popnn::lstm::basicLstmCellForwardPass(graph, prevAct, biases, prevOutput,
                                            weightsInput, weightsOutput,
                                            cellState, prog,
                                            partialsTypeStr, "");

  auto rawHostPrevAct = allocateHostMemoryForTensor(prevAct, "prevAct", graph,
                                                    tmap);
  auto rawHostBiases = allocateHostMemoryForTensor(biases, "biases", graph,
                                                   tmap);
  auto rawHostPrevOutput = allocateHostMemoryForTensor(prevOutput, "prevOutput",
                                                       graph, tmap);
  auto rawHostCellState = allocateHostMemoryForTensor(cellState, "cellState",
                                                      graph, tmap);

  std::vector<std::unique_ptr<char[]>> rawHostNextAct;
  for (auto s = 0U; s != sequenceSize; ++s) {
    rawHostNextAct.push_back(allocateHostMemoryForTensor(nextAct[s],
                                                         "nextAct" +
                                                           std::to_string(s),
                                                         graph, tmap));
  }

  Engine engine(graph, prog);

  boost::multi_array<double, 3>
      hostPrevAct(boost::extents[sequenceSize][batchSize][inputSize]);
  boost::multi_array<double, 3>
      hostWeightsOutput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [outputSize][outputSize]);
  boost::multi_array<double, 3>
      hostWeightsInput(boost::extents[BASIC_LSTM_CELL_NUM_UNITS]
                                [outputSize][inputSize]);
  boost::multi_array<double, 2>
      hostBiases(boost::extents[BASIC_LSTM_CELL_NUM_UNITS][outputSize]);
  boost::multi_array<double, 2>
      hostCellState(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      modelCellState(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 2>
      hostPrevOutput(boost::extents[batchSize][outputSize]);
  boost::multi_array<double, 3>
      hostNextAct(boost::extents[sequenceSize][batchSize][outputSize]);
  boost::multi_array<double, 3>
      modelNextAct(boost::extents[sequenceSize][batchSize][outputSize]);

  std::fill(hostPrevOutput.data(),
            hostPrevOutput.data() + hostPrevOutput.num_elements(),
            0);

  std::mt19937 randomEngine;

  writeRandomValues(hostPrevAct, -4.0, 4.0, randomEngine);
  writeRandomValues(hostPrevOutput, -3.0, 3.0, randomEngine);
  writeRandomValues(hostCellState, -3.0, 3.0, randomEngine);
  writeRandomValues(hostWeightsInput, -1.0, 1.0, randomEngine);
  writeRandomValues(hostWeightsOutput, -1.0, 1.0, randomEngine);
  writeRandomValues(hostBiases, -1.0, 1.0, randomEngine);

  modelCellState = hostCellState;

  copy(hostPrevAct, dataTypeStr, rawHostPrevAct.get());
  copy(hostCellState, dataTypeStr, rawHostCellState.get());
  copy(hostPrevOutput, dataTypeStr, rawHostPrevOutput.get());
  copy(hostBiases, dataTypeStr, rawHostBiases.get());

  for (auto u = 0U; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    boost::multi_array<double, 2> wInp = hostWeightsInput[u];
    copy(wInp, dataTypeStr, rawHostWeightsInput[u].get());
    boost::multi_array<double, 2> wOut = hostWeightsOutput[u];
    copy(wOut, dataTypeStr, rawHostWeightsOutput[u].get());
  }

  upload(engine, tmap);
  engine.run(0);    // matrix operation
  download(engine, tmap);

  copy(dataTypeStr, rawHostCellState.get(), hostCellState);

  poplib_test::lstm::basicLstmCellForwardPass(hostPrevAct, hostBiases,
                                              hostPrevOutput, hostWeightsInput,
                                              hostWeightsOutput, modelCellState,
                                              modelNextAct);

  bool matchesModel  = checkIsClose("cellState", hostCellState, modelCellState,
                                    relativeTolerance);

  for (auto s = 0U; s != rawHostNextAct.size(); ++s) {
    boost::multi_array<double, 2> subMatImp(boost::extents[batchSize]
                                                          [outputSize]);
    copy(dataTypeStr, rawHostNextAct[s].get(), subMatImp);
    boost::multi_array<double, 2> subMatRef = modelNextAct[s];
    matchesModel |= checkIsClose("nextAct", subMatRef, subMatImp,
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
