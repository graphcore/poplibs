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

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;

const poplar::OptionFlags options {
  {"debug.executionProfile", "compute_sets"}
};

//*************************************************
bool doBroadcastVectorOptimiseTest(const DeviceType &deviceType,
              const unsigned tiles,
              const Type &dataType,
              const std::vector<unsigned> &dims,
              const std::vector<unsigned> &shuffleShape,
              const unsigned dim,
              expr::BroadcastOpType operation, const bool inPlace,
              const std::function<double(double, double)> &hostFn,
              const bool doCheck,
              const bool doReport,
              const bool benchmark) {
  if(dim >= dims.size()) {
    std::cerr<<"Dim out of range: " << dim << " >= "<< dims.size() <<"\n";
    return false;
  }

  unsigned total_elems = 1;
  for(unsigned a :dims)
    total_elems = total_elems * a;

  // Test addend is the size of the specified dimension
  std::vector<double> addend(dims[dim]);
  // Program generated test data
  std::vector<double> outTest(total_elems);
  std::vector<double> inTest(total_elems);

  for (unsigned  i = 0; i < total_elems; i++) {
    inTest[i] = static_cast<double>(i) + 1;
  }

  // Initialise input pattern
  for (unsigned  i = 0; i < addend.size(); i++)
    addend[i] = static_cast<double>(i) + 1;

  //Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Create and map the tensor to produce a layout based on the shuffled
  // dimensions.  Then shuffle it to match the specified tensor shape
  std::vector<unsigned long> constructDims(dims.size());
  for(unsigned i = 0; i < dims.size(); i++)
    constructDims[i] = dims[shuffleShape[i]];
  Tensor in = graph.addVariable(dataType, constructDims, "Input Data");
  mapTensorLinearly(graph, in);

  std::vector<unsigned> invShuffleShape(dims.size());
  for(unsigned i = 0; i < dims.size(); i++)
    invShuffleShape[shuffleShape[i]] = i;
  in = in.dimShuffle(invShuffleShape);

  if(!benchmark)
    graph.createHostWrite("inStream", in, true);
  // Find the size of the total data in the innermost dimension - those after
  // the specified dimension
  unsigned innerDimsSize = 1;
  for(unsigned i = dim+1; i < dims.size();i++)
    innerDimsSize *= dims[i];

  //Input Vector
  std::vector<unsigned long> addendDims (dims.size() - dim, 1);
  addendDims[0] = addend.size();
  auto in2 = graph.addVariable(dataType, addendDims, "Vector");

  mapTensorLinearly(graph, in2);

  //Make a program sequence to run the operation
  auto prog = Sequence();
  if(inPlace) {
    if(operation == expr::BroadcastOpType::ADD)
      addInPlace(graph, in, in2, prog);
    else if(operation == expr::BroadcastOpType::MULTIPLY)
      mulInPlace(graph, in, in2, prog);
    else if(operation == expr::BroadcastOpType::SUBTRACT)
      subInPlace(graph, in, in2, prog);
  }
  else {
    if(operation == expr::BroadcastOpType::ADD)
      in = add(graph, in, in2, prog);
    if(operation == expr::BroadcastOpType::MULTIPLY)
      in = mul(graph, in, in2, prog);
    if(operation == expr::BroadcastOpType::SUBTRACT)
      in = sub(graph, in, in2, prog);
  }
  if(!benchmark) {
    graph.createHostWrite("in2Stream", in2, true);
    graph.createHostRead("outStream", in, true);
  }

  //Run each sequence and compare host and IPU result
  Engine engine(graph, prog, options);

  //Put test inputs into an array of the correct type ready to use
  std::vector<char> inTestRaw(total_elems * 4);
  std::vector<char> addendRaw(total_elems * 4);

  copy(target,inTest.data(),inTest.size(),dataType,inTestRaw.data());
  copy(target,addend.data(),addend.size(),dataType,addendRaw.data());

  std::vector<double> outHost(total_elems);
  std::vector<char> outHostRaw(total_elems * 4);
  device.bind([&](const Device &d) {
    engine.load(d);
    if(!benchmark) {
      engine.writeTensor("inStream", inTestRaw.data());
      engine.writeTensor("in2Stream", addendRaw.data());
    }
    engine.run();

    // Fetch the result and convert to a double for comparison
    if(!benchmark)
      engine.readTensor("outStream", (void*)&outHostRaw[0]);
    if(doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");

      engine.printProfileSummary(std::cout, opt);
    }

  });
  //Check the result, in the outTest array
  if(doCheck && !benchmark) {
    copy(target, dataType, outHostRaw.data(), outHost.data(), outHost.size());

    //Host generated result, start with zeros
    for(unsigned i = 0;i < total_elems ;i++)
      outTest[i] = 0;
    //Then do the operation for comparison
    unsigned dataIndex = 0;
    unsigned addendIndex = 0;
    while(dataIndex < total_elems){
      for(unsigned j = 0; j < innerDimsSize; j++) {
          outTest[dataIndex] = hostFn(inTest[dataIndex], addend[addendIndex]);
          dataIndex++;
       }
      addendIndex++;
      if(addendIndex >= addend.size())
        addendIndex = 0;
    }

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
  bool doCheck = true;
  bool doReport = false;
  bool inPlace = true;
  bool benchmark = false;
  unsigned dim = 1;
  unsigned tiles = 1;
  ShapeOption<unsigned> dims;
  ShapeOption<unsigned> shuffleShape ;
  po::options_description desc("Options");

  desc.add_options()
    ("help", "Print help")
     ("check",
     po::value<bool>(&doCheck)->default_value(doCheck),
     "Activate check for correct result")
     ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
     ("benchmark",
     po::value<bool>(&benchmark)->default_value(benchmark),
     "Execute in benchmark mode not verification mode")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type")
    ("dim",
     po::value<unsigned>(&dim)->default_value(dim),
     "Dimension")
    ("in-place",
     po::value<bool>(&inPlace)->default_value(inPlace),
     "in place operation")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("dims",
     po::value<ShapeOption<unsigned>>(&dims)->multitoken(),
     "Dimensions for scaledAdd")
     ("dim-shuffle",
      po::value<ShapeOption<unsigned>>(&shuffleShape)->multitoken(),
      "Shape to shuffle the tensor to before mapping")
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
  else {
    std::cerr<< " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  return !doBroadcastVectorOptimiseTest(deviceType, tiles, dataType, dims.val,
                  shuffleShape.val, dim,
                  broadcastOperation, inPlace,
                  broadcastHostFn, doCheck, doReport, benchmark);
}
