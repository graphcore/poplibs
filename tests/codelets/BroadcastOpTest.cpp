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
              bool testSupervisor,
              unsigned bElems,
              bool inPlace,
              const std::function<double(double, double)> &hostFn,
              bool doCheck,
              bool doReport) {

  // Whole data array size
  auto total_elems = rows * columns;

  // Program generated test data
  std::vector<double> outTest(total_elems);
  std::vector<double> inTest(total_elems);
  std::vector<double> BTest(bElems);

  // Initialise input patterns
  for (unsigned  i = 0; i < total_elems; i++)
    inTest[i] = static_cast<double>(i) + 1;

  double k = 4;
  for (unsigned  i = 0; i < BTest.size(); i++)
    BTest[i] = static_cast<double>(i) + k;

  //Create Graph object, target and device
  auto device = createTestDevice(deviceType);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  const auto vectorWidth = target.getVectorWidth(dataType);

  Tensor in;
  if (testSupervisor)
    in = graph.addVariable(dataType, {total_elems}, "Input Data");
  else
    in = graph.addVariable(dataType, {rows, columns}, "Input Data");

  graph.setTileMapping(in, 0);

  // Create B as scalar or vector, as required
  Tensor B;
  if (bElems==1)
    B = graph.addVariable(dataType, {}, "Constant");
  else
    B = graph.addVariable(dataType, {bElems}, "Constant");
  graph.setTileMapping(B, 0);

  // Output Tensor, used only if not in-place
  Tensor out;
  if (!inPlace) {
    if (testSupervisor)
      out = graph.addVariable(dataType, {total_elems}, "Output Data");
    else
      out = graph.addVariable(dataType, {rows, columns}, "Output Data");
    graph.setTileMapping(out, 0);
  }

  // Make a sequence to run the operation
  Sequence sequence;
  ComputeSet testComputeSet=graph.addComputeSet("computeOp");
  std::string vertexName, vertexClass;

  // There are 8 (counting the "InPlace" options) vertex variants to test,
  // named as follows::
  //
  // If 'B' has 1 element (i.e. a scalar or a 1-elem tensor):
  //   "popops::BroadcastScalar1D[InPlace]Supervisor"    :'data' is 1D
  //   "popops::BroadcastScalar2DData[InPlace]"          :'data' is 2D
  //
  // If 'B' is a vector:
  //   "popops::BroadcastVectorOuter[InPlace]Supervisor" :'data' is 2D flattened
  //   "popops::Broadcast2D[InPlace]"                    :'data' is 2D
  //
  if (testSupervisor) {
    if (bElems==1) {
      vertexName = inPlace ? "popops::BroadcastScalar1DInPlaceSupervisor"
                           : "popops::BroadcastScalar1DSupervisor";
    }
    else {
      if (columns % vectorWidth == 0) {
        vertexName = inPlace ?
                     "popops::BroadcastVectorOuterByRowInPlaceSupervisor"
                   : "popops::BroadcastVectorOuterByRowSupervisor";
      } else {
        vertexName = inPlace ?
                     "popops::BroadcastVectorOuterByColumnInPlaceSupervisor"
                   : "popops::BroadcastVectorOuterByColumnSupervisor";
      }
    }
  }
  else {
    if (bElems==1) {
      vertexName = inPlace ? "popops::BroadcastScalar2DDataInPlace"
                           : "popops::BroadcastScalar2DData";
    }
    else {
      vertexName = inPlace ? "popops::BroadcastScalar2DInPlace"
                           : "popops::BroadcastScalar2D";
    }
  }

  vertexClass = templateVertex(vertexName, operation, dataType);

  auto vertex = graph.addVertex(testComputeSet, vertexClass);
  graph.setTileMapping(vertex, 0);

  graph.connect(vertex["data"],in);

  if (!inPlace) {
    graph.connect(vertex["out"], out);
  }

  graph.connect(vertex["B"], B);

  if (vertexName.find("VectorOuter")!=std::string::npos) {
    graph.setInitialValue(vertex["columns"], columns);
    graph.setInitialValue(vertex["rows"], rows);
  }


  //allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto input = allocateHostMemoryForTensor(in,"in",graph, uploadProg,
                                                downloadProg, tmap);
  auto inputB = allocateHostMemoryForTensor(B,"inB",graph, uploadProg,
                                                 downloadProg, tmap);

  sequence.add(Execute(testComputeSet));

  // If in-place, 'in' will contain the result
  graph.createHostRead("outStream", inPlace? in : out);

  //Run sequence and compare host and IPU result
  Engine engine(graph, Sequence(uploadProg, sequence, downloadProg), options);
  attachStreams(engine, tmap);

  //Put test inputs into an array of the correct type ready to use
  copy(target, inTest.data(), inTest.size(), dataType, input.get());
  copy(target, BTest.data(), BTest.size(), dataType, inputB.get());

  std::vector<double> outHost(total_elems);
  std::vector<char> outHostRaw(total_elems * 4);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");

      engine.printProfileSummary(std::cerr, opt);
    }

    // Fetch the result and convert to a double for comparison
    engine.readTensor("outStream", (void*)&outHostRaw[0]);
  });

  copy(target, dataType, outHostRaw.data(), outHost.data(), outHost.size());

   //Host generated result, start with zeros
  for (unsigned i = 0; i < total_elems; i++)
      outTest[i] = 0;
  //Then do the operation for comparison
  unsigned bIndex = 0;
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < columns; j++) {
      if (bElems==1) {
        outTest[j + i * columns] = hostFn(inTest[j + i * columns], BTest[0]);
      }
      else {
        outTest[j + i * columns] = hostFn(inTest[j + i * columns],
                                   BTest[bIndex]);
       }
    }
    bIndex++;
    if(bIndex == bElems) {
      bIndex = 0;
    }
  }
  //Check the result, in the outTest array
  if (doCheck) {
    bool check = checkIsClose("StdTest",
        outHost.data(), {outHost.size()}, outTest.data(), outTest.size(),
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
  unsigned bLength = 1;
  bool testSupervisor = false;
  bool inPlace = true;

  po::options_description desc("Options");

  desc.add_options()
    ("help", "Print help")
     ("check",
     po::value<bool>(&doCheck)->default_value(doCheck),
     "Activate check for correct result")
     ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
     ("b-length",
     po::value<unsigned>(&bLength)->default_value(bLength),
     "Length of second tensor")
     ("supervisor",
     po::value<bool>(&testSupervisor)->default_value(testSupervisor),
     "Test supervisor vertices")
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
    ("in-place",
     po::value<bool>(&inPlace)->required(),
     "Test the in-place variant")
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
  }
  else {
    std::cerr<< " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  if (!doBroadcastOpTest(deviceType, dataType, rows, columns,
          broadcastOperation, testSupervisor, bLength, inPlace,
          broadcastHostFn, doCheck, doReport))
    return 1;

  return 0;
}
