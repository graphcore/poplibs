// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the elementwise vertex operations.
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
  {"debug.executionProfile", "vertices"}
};

//*************************************************
bool doUnaryOpTest(const DeviceType &deviceType,
              const Type &dataType,
              const Type &dataTypeOut,
              unsigned rows,
              unsigned columns,
              expr::UnaryOpType operation,
              const std::function<double(double)> &hostFn,
              unsigned inPlace,
              bool doCheck,
              bool doReport) {

  // Whole data array size
  auto total_elems = rows * columns;
  auto total_size = rows * columns;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_elems);

  // Initialise input pattern, account for integers being tested
  for (unsigned  i = 0; i < total_elems; i++)
    inTest[i] = i + 1;

  //Create Graph object, target and device
  Device device = createTestDevice(deviceType);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  //Input data
  Tensor in = graph.addVariable(dataType,{rows, columns}, "Input Data");
  graph.setTileMapping(in,0);

  //Result data
  Tensor out = graph.addVariable(dataTypeOut,{rows, columns}, "Output");
  graph.setTileMapping(out,0);

  //allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto input = allocateHostMemoryForTensor(in,"in",graph,uploadProg,
                                                downloadProg,tmap);

  //Make a sequence to zero output memory and run the operation
  Sequence sequence;
  ComputeSet testComputeSet = graph.addComputeSet("computeOp");
  std::string vertexClass;
  if(inPlace){
    vertexClass = templateVertex(rows > 1 ?
                                        "popops::UnaryOp2DInPlace" :
                                        "popops::UnaryOp1DInPlaceSupervisor",
                                        operation,dataType);
  }
  else {
    vertexClass = templateVertex(rows > 1 ?
                                        "popops::UnaryOp2D" :
                                        "popops::UnaryOp1DSupervisor",
                                        operation,dataType);
  }
  auto vertex = graph.addVertex(testComputeSet,vertexClass);
  graph.setTileMapping(vertex,0);

  if(inPlace) {
    if(rows == 1){
      graph.connect(vertex["inOut"],in.reshape({columns}));
    }
    else {
      graph.connect(vertex["inOut"],in);
    }
  }
  else {
    if(rows == 1){
      graph.connect(vertex["in"],in.reshape({columns}));
      graph.connect(vertex["out"],out.reshape({columns}));
    }
    else {
      graph.connect(vertex["in"],in);
      graph.connect(vertex["out"],out);

    }
  }
  if(dataTypeOut != BOOL)
    popops::zero(graph,out,sequence,"Zero output");
  sequence.add(Execute(testComputeSet));
  if(inPlace)
    graph.createHostRead("outStream", in);
  else
    graph.createHostRead("outStream", out);
  //Run each sequence and compare host and IPU result
  Engine engine(graph,Sequence(uploadProg, sequence, downloadProg), options);
  engine.load(device);
  attachStreams(engine, tmap);

  //Put test inputs into an array of the correct type ready to use
  copy(target,inTest.data(),inTest.size(),dataType,input.get());

  engine.run(0);

  if(doReport) {
    OptionFlags opt;
    opt.set("doLayerWiseBreakdown", "true");

    auto execReport = engine.getExecutionReport(opt);
    auto graphReport = engine.getGraphReport(opt);
    graphReport.printSummary(std::cerr);
    execReport.printSummary(std::cerr);
  }

  // Fetch the result and convert to a double for comparison
  std::vector<double> outHost(total_size );
  std::vector<char> outHostRaw(total_size*4 );

  engine.readTensor("outStream", (void*)&outHostRaw[0]);
  copy(target, dataTypeOut, outHostRaw.data(), outHost.data(), outHost.size());

  //Host generated result, start with zeros
  for(unsigned i = 0;i<total_size ;i++)
      outTest[i] = 0;
  //Then do the operation for comparison
  for(unsigned i = 0; i<rows; i++) {
    for(unsigned j = 0; j<columns; j++) {
      outTest[j + i * columns] = hostFn(inTest[j + i * columns]);
    }
  }
  //Check the result, in the outTest array
  if(doCheck){
    bool check = checkIsClose("StdTest",
        outHost.data(), {outHost.size()}, outTest.data(), outTest.size(),
        0.05,0.05);
    return check;
  }
  else {
    return true;
  }
}

//*************************************************
bool doBinaryOpTest(const DeviceType &deviceType,
              const Type &dataType,
              const Type &dataTypeOut,
              unsigned rows,
              unsigned columns,
              expr::BinaryOpType operation,
              const std::function<double(double, double)> &hostFn,
              unsigned inPlace,
              bool doCheck,
              bool doReport) {

  // Whole data array size
  auto total_elems = rows * columns;
  auto total_size = rows * columns;

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> in1Test(total_elems);
  std::vector<double> in2Test(total_elems);

  // Initialise input pattern, account for ints, unsigned ints being tested
  int factor = 1;
  for (unsigned  i = 0; i < total_elems; i++)
          in1Test[i] = static_cast<double>(i) + 1;
  for (unsigned  i = 0; i < total_elems; i++)
  {
          in2Test[i] = static_cast<double>(i)* 2 * factor;
          if(dataType != UNSIGNED_INT) {
            if(!(i&4))
              factor = factor * -1;
          }
  }
  //Create Graph object, target and device
  Device device = createTestDevice(deviceType);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  //Input data
  Tensor in1=graph.addVariable(dataType,{rows, columns}, "Input Data 1");
  Tensor in2=graph.addVariable(dataType,{rows, columns}, "Input Data 2");
  graph.setTileMapping(in1,0);
  graph.setTileMapping(in2,0);

  //Result data
  Tensor out=graph.addVariable(dataTypeOut,{rows, columns}, "Output");
  graph.setTileMapping(out,0);

  //allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char*>> tmap;
  auto input1 = allocateHostMemoryForTensor(in1,"in1",graph,uploadProg,
                                                downloadProg,tmap);

  auto input2 = allocateHostMemoryForTensor(in2,"in2",graph,uploadProg,
                                                downloadProg,tmap);

  //Make a sequence to zero output memory and run the operation
  Sequence sequence;
  ComputeSet testComputeSet=graph.addComputeSet("computeOp");
  std::string vertexClass;
  if(inPlace){
    vertexClass = templateVertex(rows > 1 ?
                                        "popops::BinaryOp2DInPlace" :
                                        "popops::BinaryOp1DInPlaceSupervisor",
                                        operation,dataType);
  }
  else {
    vertexClass = templateVertex(rows > 1 ?
                                        "popops::BinaryOp2D" :
                                        "popops::BinaryOp1DSupervisor",
                                        operation,dataType);
  }
  auto vertex = graph.addVertex(testComputeSet,vertexClass);
  graph.setTileMapping(vertex,0);

  if(inPlace) {
    if(rows == 1){
      graph.connect(vertex["in1Out"],in1.reshape({columns}));
      graph.connect(vertex["in2"],in2.reshape({columns}));
    }
    else {
      graph.connect(vertex["in1Out"],in1);
      graph.connect(vertex["in2"],in2);
    }
  }
  else {
    if(rows == 1){
      graph.connect(vertex["in1"],in1.reshape({columns}));
      graph.connect(vertex["in2"],in2.reshape({columns}));
      graph.connect(vertex["out"],out.reshape({columns}));
    }
    else {
      graph.connect(vertex["in1"],in1);
      graph.connect(vertex["in2"],in2);
      graph.connect(vertex["out"],out);

    }
  }
  if(dataTypeOut != BOOL)
    popops::zero(graph,out,sequence,"Zero output");
  sequence.add(Execute(testComputeSet));

  if(inPlace)
    graph.createHostRead("outStream", in1);
  else
    graph.createHostRead("outStream", out);

  //Run each sequence and compare host and IPU result
  Engine engine(graph,Sequence(uploadProg, sequence, downloadProg), options);
  engine.load(device);
  attachStreams(engine, tmap);

  //Put test inputs into an array of the correct type ready to use
  copy(target,in1Test.data(),in1Test.size(),dataType,input1.get());
  copy(target,in2Test.data(),in2Test.size(),dataType,input2.get());

  engine.run(0);

  if(doReport) {
    OptionFlags opt;
    opt.set("doLayerWiseBreakdown", "true");

    auto execReport = engine.getExecutionReport(opt);
    auto graphReport = engine.getGraphReport(opt);
    graphReport.printSummary(std::cerr);
    execReport.printSummary(std::cerr);
  }

  // Fetch the result and convert to a double for comparison
  std::vector<double> outHost(total_size);
  std::vector<char> outHostRaw(total_size*4);
  engine.readTensor("outStream", (void*)&outHostRaw[0]);
  copy(target, dataTypeOut, outHostRaw.data(), outHost.data(), outHost.size());

   //Host generated result, start with zeros
  for(unsigned i = 0;i<total_size ;i++)
      outTest[i] = 0;
  //Then do the operation for comparison
  for(unsigned i = 0; i<rows; i++) {
      for(unsigned j = 0; j<columns; j++) {
          outTest[j + i * columns] = hostFn( in1Test[j + i * columns],
                                             in2Test[j + i * columns]);
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
  unsigned rows, columns, inPlace, unaryOp;
  bool doCheck = true;
  bool doReport = false;
  bool outputBool = false;

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
    ("in-place",
     po::value<unsigned>(&inPlace)->required(),
     "In Place")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Allowed operations:\n"
     "  Unary: EXPONENT IS_FINITE LOGARITHM LOGARITHM_ONE_PLUS NEGATE\n"
     "         SIGNUM SIN SQRT SQUARE TANH SIGMOID RSQRT\n"
     "  Binary:ADD ATAN2 DIVIDE EQUAL GREATER_THAN\n"
     "         MULTIPLY MAXIMUM POWER REMAINDER")
    ;
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
  expr::BinaryOpType binaryOperation;
  expr::UnaryOpType unaryOperation;
  std::function<double(double, double)> binaryHostFn;
  std::function<double(double)> unaryHostFn;

  // Unary operations
  if(operation == "EXPONENT") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::EXPONENT;
    unaryHostFn = [](double x) -> double {
          return std::exp(x);};
  }
  else if(operation == "IS_FINITE") {
    unaryOp = 1;
    outputBool = true;
    unaryOperation = expr::UnaryOpType::IS_FINITE;
    unaryHostFn = [](double x) -> double {
      return std::isfinite(x);};
  }
  else if(operation == "LOGARITHM") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::LOGARITHM;
    unaryHostFn = [](double x) -> double {
          return std::log(x);};
  }
  else if(operation == "LOGARITHM_ONE_PLUS") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::LOGARITHM_ONE_PLUS;
    unaryHostFn = [](double x) -> double {
          return std::log1p(x);};
  }
  else if(operation == "NEGATE") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::NEGATE;
    unaryHostFn = [](double x) -> double {
          return -1 * x;};
  }
  else if(operation == "SIGNUM") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::SIGNUM;
    unaryHostFn = []( double x) -> double {
          return (0 < x) - (x < 0);};
  }
  else if(operation == "SIN") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::SIN;
    unaryHostFn = [](double x) -> double {
          return std::sin(x);};
  }
  else if(operation == "SQRT") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::SQRT;
    unaryHostFn = [](double x) -> double {
          return std::sqrt(x);};
  }
  else if(operation == "SQUARE") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::SQUARE;
    unaryHostFn = [](double x) -> double {
          return (x * x);};
  }
  else if(operation == "TANH") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::TANH;
    unaryHostFn = [](double x) -> double {
          return std::tanh(x);};
  }
  else if(operation == "SIGMOID") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::SIGMOID;
    unaryHostFn = [](double x) -> double {
      return 1.0 / (1.0 + std::exp(-x));
    };
  }
  else if(operation == "RSQRT") {
    unaryOp = 1;
    unaryOperation = expr::UnaryOpType::RSQRT;
    unaryHostFn = [](double x) -> double {
      return 1.0 / std::sqrt(x);
    };
  }

  // Binary operations
  else if(operation == "ADD") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::ADD;
    binaryHostFn = [](double x, double y) -> double {
          return x + y;};
  }
  else if(operation == "ATAN2") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::ATAN2;
    binaryHostFn = [](double x, double y) -> double {
          return std::atan2(x, y);};
  }
  else if(operation == "DIVIDE") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::DIVIDE;
    binaryHostFn = [](double x, double y) -> double {
          return x / y;};
  }
  else if(operation == "EQUAL") {
    unaryOp = 0;
    outputBool = true;
    binaryOperation = expr::BinaryOpType::EQUAL;
    binaryHostFn = [](double x, double y) -> double {
          return x == y;};
  }
  else if(operation == "GREATER_THAN") {
    unaryOp = 0;
    outputBool = true;
    binaryOperation = expr::BinaryOpType::GREATER_THAN;
    binaryHostFn = [](double x, double y) -> double {
          return x > y;};
  }
  else if(operation == "MAXIMUM") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::MAXIMUM;
    binaryHostFn = [](double x, double y) -> double {
          return std::max(x, y);};
  }
  else if(operation == "MULTIPLY") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::MULTIPLY;
    binaryHostFn = [](double x, double y) -> double {
          return x * y;};
  }
  else if(operation == "POWER") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::POWER;
    binaryHostFn = [](double x, double y) -> double {
          return pow(x, y);};
  }
  else if(operation == "REMAINDER") {
    unaryOp = 0;
    binaryOperation = expr::BinaryOpType::REMAINDER;
    binaryHostFn = [](double x, double y) -> double {
          return fmod(x,y);};
  }
  else {
    std::cerr<< " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  Type dataTypeOut;
  if(outputBool)
    dataTypeOut = BOOL;
  else
    dataTypeOut = dataType;

  if(unaryOp) {
   if(!doUnaryOpTest(deviceType, dataType, dataTypeOut, rows, columns,
                    unaryOperation, unaryHostFn, inPlace, doCheck, doReport))
      return 1;

  }
  else {
    if(!doBinaryOpTest(deviceType, dataType, dataTypeOut, rows, columns,
                    binaryOperation, binaryHostFn, inPlace, doCheck, doReport))
      return 1;
  }
  return 0;
}
