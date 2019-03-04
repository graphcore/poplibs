// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the broadcastOp vertex operations.
// Used to verify aspects of implementation that
// aren't simply to correctness of arithmetic on a single item. Also
// for benchmarking.
// Eg - different length vectors for Supervisor vertices or other
// vectorised implementations, where data quantity is important.
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include "popops/ElementWise.hpp"
#include "../lib/popops/ExprOpUtil.hpp"
#include <poplibs_test/Util.hpp>

#include <boost/program_options.hpp>
#include <iostream>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;

const poplar::OptionFlags options {
  {"target.workerStackSizeInBytes", "0x1000"},
  {"debug.executionProfile", "compute_sets"}
};

//*************************************************
bool doBroadcastOpTest(const DeviceType &deviceType,
              const Type &dataType,
              unsigned rows,
              unsigned columns,
               expr::BroadcastOpType operation,
              const std::function<double(double, double)> &hostFn,
              bool doCheck,
              bool doReport) {

  // Whole data array size
  auto total_elems = rows * columns;

  // Program generated test data
  std::vector<double> outTest(total_elems);
  std::vector<double> inTest(total_elems);

  // Initialise input pattern
  for (unsigned  i = 0; i < total_elems; i++)
          inTest[i] = static_cast<double>(i) + 1;
  double k = 3;

  //Create Graph object, target and device
  auto device = createTestDevice(deviceType);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  //Input / result data, operations are all in-place
  Tensor in = graph.addVariable(dataType,{rows, columns}, "Input Data");
  graph.setTileMapping(in,0);

  //allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto input = allocateHostMemoryForTensor(in,"in",graph,uploadProg,
                                                downloadProg,tmap);

  //Make a sequence to run the operation
  Sequence sequence;
  ComputeSet testComputeSet=graph.addComputeSet("computeOp");
  std::string vertexClass;

  vertexClass = templateVertex(rows > 1 ? "popops::BroadcastOp2DInPlace" :
                                    "popops::BroadcastOp1DInPlaceSupervisor",
                                    operation, dataType);
  auto vertex = graph.addVertex(testComputeSet,vertexClass);
  graph.setTileMapping(vertex,0);


  Tensor B = graph.addVariable(dataType, {}, "Constant");
  graph.connect(vertex["B"], B);
  graph.setTileMapping(B, 0);
  graph.setInitialValue(B, k);


  graph.setTileMapping(vertex, 0);
  if(rows == 1){
    graph.connect(vertex["data"],in.reshape({columns}));
  }
  else {
    graph.connect(vertex["data"],in);
  }
  sequence.add(Execute(testComputeSet));

  graph.createHostRead("outStream", in);

  //Run each sequence and compare host and IPU result
  Engine engine(graph,Sequence(uploadProg, sequence, downloadProg), options);
  attachStreams(engine, tmap);

  //Put test inputs into an array of the correct type ready to use
  copy(target,inTest.data(),inTest.size(),dataType,input.get());

  std::vector<double> outHost(total_elems);
  std::vector<char> outHostRaw(total_elems * 4);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);

    if(doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");

      engine.printProfileSummary(std::cerr, opt);
    }

    // Fetch the result and convert to a double for comparison
    engine.readTensor("outStream", (void*)&outHostRaw[0]);
  });

  copy(target, dataType, outHostRaw.data(), outHost.data(), outHost.size());

   //Host generated result, start with zeros
  for(unsigned i = 0;i<total_elems ;i++)
      outTest[i] = 0;
  //Then do the operation for comparison
  for(unsigned i = 0; i<rows; i++) {
      for(unsigned j = 0; j<columns; j++) {
          outTest[j + i * columns] = hostFn( inTest[j + i * columns], k);
       }
  }
  //Check the result, in the outTest array
  if(doCheck) {
    bool check = checkIsClose("StdTest",
        outHost.data(),{outHost.size()},outTest.data(),outTest.size(),
        0.05,0.05);
    return check;
  }
  else {
    return true;
  }
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  std::string operation;
  unsigned rows, columns;
  bool doCheck = true;
  bool doReport = false;

  po::options_description desc("Options");

  desc.add_options()
    ("help", "Print help")
     ("check",
     po::value<bool>(&doCheck)->default_value(doCheck),
     "Activate check for correct result")
     ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type")
    ("rows",
     po::value<unsigned>(&rows)->required(),
     "In/Out data rows")
    ("columns",
     po::value<unsigned>(&columns)->required(),
     "In/Out data columns")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Allowed operations: ADD MULTIPLY SUBTRACT VARIANCE_TO_INV_STD_DEV"
     " INV_STD_DEV_TO_VARIANCE\n");
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
  expr::BroadcastOpType broadcastOperation;
  std::function<double(double, double)> broadcastHostFn;

  // Operations
if(operation == "ADD") {
    broadcastOperation = expr::BroadcastOpType::ADD;
    broadcastHostFn = [](double x, double y) -> double {
          return x + y;};
  }
  else if(operation == "MULTIPLY") {
    broadcastOperation = expr::BroadcastOpType::MULTIPLY;
    broadcastHostFn = [](double x, double y) -> double {
          return x * y;};
  }
  else if(operation == "SUBTRACT") {
    broadcastOperation = expr::BroadcastOpType::SUBTRACT;
    broadcastHostFn = [](double x, double y) -> double {
          return x - y;};
  }
  else if(operation == "INV_STD_DEV_TO_VARIANCE") {
    broadcastOperation = expr::BroadcastOpType::INV_STD_DEV_TO_VARIANCE;
    broadcastHostFn = [](double x, double y) -> double {
          return (1/(x * x)) - y;};
  }
  else if(operation == "VARIANCE_TO_INV_STD_DEV") {
    broadcastOperation = expr::BroadcastOpType::VARIANCE_TO_INV_STD_DEV;
    broadcastHostFn = [](double x, double y) -> double {
          return 1/sqrt(x+y);};
  }else {
    std::cerr<< " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  if(!doBroadcastOpTest(deviceType, dataType, rows, columns,
                  broadcastOperation, broadcastHostFn, doCheck, doReport))
    return 1;

  return 0;
}
