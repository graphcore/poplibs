// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for generating a broadcastOp pattern.
// Used to generate customised patterns that may require optimised vertices.
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include "popops/ElementWise.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/program_options.hpp>

#include <exception>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;

const poplar::OptionFlags options{{"debug.instrumentCompute", "true"}};

//******************************************************************************
//
// Linear algebra dictates that an elementwise binary operation can be performed
// on tensors if and only if they share the same dimensions exactly. However for
// efficient storage and execution the component tensors may be stored with
// differing dimensions provided they satisfy Python Numpy broadcast rules. In
// this scenario, some of the entries of the tensor of lower dimensions will
// need to be duplicated in order to form the missing dimension(s), before the
// elementise binary operation can be performed. This process is known as
// "broadcasting".
//
// For example, suppose a binary operation needs to be peformed between tensors
// A and B, where B has fewer dimensions than A, but satisfies the Python numpy
// broadcasting rules.
//
// A simple method for broadcasting tensor B would be to duplicate values at
// specific indices or sequences of indices of B to create a new tensor
// B' in memory which is laid out in the same order as the tensor A. The mapping
// of indices from B to B' forms a pattern. Note that the pattern is formed with
// the indices into tensor B and not the actual contents of B. Depending on the
// pattern formed, there may exist a more efficient broadcast method that
// altogether avoids the intermediate step of creating tensor B'.
//
// Broadcast patterns involve a recurring sequence of indices called a "motif".
// At the lowest level a pattern is formed by contiguous runs of identical
// indices. Each such run is called a "dash". Any broadcast pattern can be
// described using the following triple of integers:
//
//     (innerFactor, motifLength, outerFactor)
//
//  where innerFacter (iF) : the number of index repetitions in a dash.
//        motifLength (mL) : the number of dash repetitions.
///       outerFactor (oF) : the number of motif repetitions.
//
// For example the following pattern of indices is formed at the lowest level by
// two element long dash sections. Three consecutive dash sections form a
// motif and there are four motifs in total. Hence the pattern is completely
// specifiedf by the triple "(iF=2, mL=3, oF=4)".
//
//     0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,2,2
//                            |   |   |   |
//                Dashes ---> |-1-|-2-|-3-|
//                             -----------
//                                  |
//                                motif
//
//
// Pattern Generation
// -------------------
//
// The possible patterns can be classified as follows:
//
// Case 1: Patterns of the form (iF, mL, 1)
//
//         This set of patterns can be created using a tensor A of dimension
//         (mL,iF) and a column vector tensor B of dimension (mL,1).
//
//         Suppose mL=3, iF=4, A=[[ a0, a1, a2, a3],      B=[[ b0],
//                                [ a4, a5, a6, a7],         [ b1],
//                                [ a8, a9,a10,a11]]         [ b2]]
//                                                  3x4            3x1
//
//         where the subscripts in each tensor entry describes the order in
//         which these entries are stored sequentially in memory.
//
//         The broadcasted tensor B'=[[ b0, b0, b0, b0], which is stored in
//                                    [ b1, b1, b1, b1],
//                                    [ b2, b2, b2, b2]]
//
//         memory as "b0, b0, b0, b0, b1, b1, b1, b1, b2, b2, b2, b2" which is
//         defined by triple (4,3,1).
//
// Case 2: Patterns of the form (1, mL, oF)
//
//         This set of patterns can be created using a tensor A of dimension
//         (oF,mL) and a row vector tensor B of dimension (1,mL).
//
//         Suppose oF=3, mL=4, A=[[ a0, a1, a2, a3],   B=[[ b0, b1, b2, b3]]
//                              [ a4, a5, a6, a7],                       1x4
//                              [ a8, a9,a10,a11]]
//                                                3x4
//
//         where the subscripts in each tensor entry describes the order in
//         which these entries are stored sequentially in memory.
//
//         The broadcasted tensor B'=[[ b0, b1, b2, b3], which is stored in
//                                    [ b0, b1, b2, b3],
//                                    [ b0, b1, b2, b3]]
//
//         memory as "b0, b1, b2, b3, b0, b1, b2, b3, b0, b1, b2, b3" which is
//         defined by triple (1,4,3).
//
// Case 3 - The Specification of a pattern in General:
//         Patterns of the form (iF, mL, oF). This is the general case. The
//         previous two cases are particular special cases of this case.
//
//         These set of patterns can be created using tensor A of dimension
//         (oF,mL,iF) and tensor B of dimension (1,mL,1).
//
//         Suppose oF=2, mL=3, iF=4, A=[[[ a0, a1, a2, a3],
//                                       [ a4, a5, a6, a7],
//                                       [ a8, a9,a10,a11]],
//
//                                      [[a12,a13,a14,a15],
//                                       [a16,a17,a18,a19],
//                                       [a20,a21,a22,a23]]]
//                                                          2x3x4
//
//                                   B=[[[ b0],
//                                       [ b1],
//                                       [ b2]]]
//                                             1x3x1
//
//         where the subscripts in each tensor entry describes the order in
//         which these entries are stored sequentially in memory.
//
//         The broadcasted tensor B'=[[[ b0, b0, b0, b0], which is stored in
//                                     [ b1, b1, b1, b1],
//                                     [ b2, b2, b2, b2]],
//
//                                    [[ b0, b0, b0, b0],
//                                     [ b1, b1, b1, b1],
//                                     [ b2, b2, b2, b2]]]
//
//         memory as "b0, b0, b0, b0, b1, b1, b1, b1, b2, b2, b2, b2, b0, b0,
//         b0, b0, b1, b1, b1, b1, b2, b2, b2, b2," and is
//         defined by triple (4,3,2).
//
// Case 4 - Degenerate case:
//         Patterns of the form "(iF, 1, oF) where oF > 1" cannot be created,
//         as they can always be expressed using the alternate pattern
//         "(iF*oF, 1, 1)".
//
// The implementation is based on the general specification described in Case 3.
// The special cases 1 and 2 are described only for the purpose of simplifying
// the explanation.
//
// Multiple patterned regions per tile are achieved as follows:
//  1.  Generate multiple copies of the requested pattern, which can be done
//      by adding dimensions. Two additional dimensions are used for this
//      purpose. The length of the first additional dimension is set to the
//      requested number of regions per tile. The length of the second
//      additional dimension is set to the number of tiles requested.
//
//        - Using this scheme, more than one regions per tile cannot be
//          created by using less than two tiles.
//
//  2.  Slice the 5-dimension tensor along the second dimension (i.e., dimension
//      index 1) and create 'tiles' number of equal slices. Map each slice to a
//      different tile.
//
//  The 5-dimension tensor has the following dimensions:
//
//     Dimension Index |   Description
//     ----------------+-----------------------------
//            0        | Number of regions per tile
//            1        | Number of tiles
//            2        | Output Factor
//            3        | Motif Length
//            4        | Inner Factor
//
//******************************************************************************
bool doBroadcastVectorOptimiseTest(
    const DeviceType &deviceType, const unsigned tiles, const Type &dataType,
    const unsigned innerFactor, unsigned motifLength, unsigned outerFactor,
    const unsigned regionsPerTile, expr::BroadcastOpType operation,
    const bool inPlace, const std::function<double(double, double)> &hostFn,
    const bool doReport, const bool doPrintTensors, const bool ignoreData,
    bool enableOptimisations) {
  const unsigned dimPatternOffset = 2;
  const unsigned dimMotifOffset = 3;

  // Work out the dimensions of Tensor A
  const std::vector<unsigned long> dims = {regionsPerTile, tiles, outerFactor,
                                           motifLength, innerFactor};

  // Work out the dimensions of Tensor B
  const std::vector<std::size_t> in2Dims = {regionsPerTile, tiles, 1,
                                            motifLength, 1};

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Create tensor 'in' which includes tensor A encapsulated within two
  // additional dimensions if multiple patterned regions are required.
  Tensor in = graph.addVariable(dataType, dims, "Input Data");

  // Create tensor 'in2' which includes tensor B encapsulated within two
  // additional dimensions if multiple patterned regions are required.
  // Element-wise ops between tensor A and vector B need to involve a vector
  // broadcast.
  Tensor in2 = graph.addVariable(dataType, in2Dims, "vector");

  // Generate test data for Tensor A
  std::vector<double> inHost(in.numElements());

  for (unsigned i = 0; i < in.numElements(); i++) {
    inHost[i] = static_cast<double>(i) + 1;
  }

  // Generate test data for Tensor B
  std::vector<double> in2Host(in2.numElements());

  for (unsigned i = 0; i < in2Host.size(); i++) {
    in2Host[i] = static_cast<double>(i) + 1;
  }

  // If multiple patterned regions are required, map a different slice of
  // these patterns to each tile
  for (unsigned i = 0; i < tiles; i++) {
    graph.setTileMapping(in.slice(Interval(i, i + 1), 1), i);
    graph.setTileMapping(in2.slice(Interval(i, i + 1), 1), i);
  }

  OptionFlags opOpts{{"enableVectorBroadcastOptimisations",
                      (enableOptimisations ? "true" : "false")}};

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if (inPlace) {
    switch (operation) {
    case expr::BroadcastOpType::ADD:
      addInPlace(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BroadcastOpType::MULTIPLY:
      mulInPlace(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BroadcastOpType::SUBTRACT:
      subInPlace(graph, in, in2, prog, "", opOpts);
      break;
    default:
      throw std::logic_error("Unrecognised operation type!");
    }
    out = in;
  } else {
    switch (operation) {
    case expr::BroadcastOpType::ADD:
      out = add(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BroadcastOpType::MULTIPLY:
      out = mul(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BroadcastOpType::SUBTRACT:
      out = sub(graph, in, in2, prog, "", opOpts);
      break;
    default:
      throw std::logic_error("Unrecognised operation type!");
    }
  }

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> inHostRaw;
  std::unique_ptr<char[]> in2HostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  if (!ignoreData) {
    inHostRaw = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                            downloadProg, tmap);
    in2HostRaw = allocateHostMemoryForTensor(in2, "in2", graph, uploadProg,
                                             downloadProg, tmap);
    if (!inPlace) {
      outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                               downloadProg, tmap);
      outHostRawPtr = outHostRaw.get();
    } else {
      outHostRawPtr = inHostRaw.get();
    }
  }

  if (doPrintTensors) {
    prog.add(PrintTensor("in", in));
    prog.add(PrintTensor("in2", in2));
    if (!inPlace)
      prog.add(PrintTensor("out", out));
  }

  // Run sequences and compare host and IPU result
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  attachStreams(engine, tmap);

  if (!ignoreData) {
    copy(target, inHost.data(), inHost.size(), dataType, inHostRaw.get());
    copy(target, in2Host.data(), in2Host.size(), dataType, in2HostRaw.get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
  });

  // Check the result
  if (!ignoreData) {
    std::vector<double> outHost(in.numElements());
    copy(target, dataType, outHostRawPtr, outHost.data(), outHost.size());

    std::vector<double> outModel(in.numElements());
    for (unsigned i = 0; i < in.numElements(); i++)
      outModel[i] = 0;

    /* Calculate the number of elements in a pattern, i.e,. tensor A */
    auto patternDimsSize =
        std::accumulate(&dims[dimPatternOffset], &dims[dims.size()],
                        std::size_t(1), std::multiplies<std::size_t>());

    /* Calculate the number of element in tensor B */
    auto bVectorLength = in2Dims[dimMotifOffset];
    for (std::size_t count = 0; count < (in.numElements() / patternDimsSize);
         ++count) {
      auto inOffset = count * patternDimsSize;
      auto in2Offset = count * bVectorLength;
      for (std::size_t i = 0; i < patternDimsSize; ++i) {
        auto in2Index = (i / innerFactor) % bVectorLength;
        outModel[inOffset + i] =
            hostFn(inHost[inOffset + i], in2Host[in2Offset + in2Index]);
      }
    }

    return checkIsClose("StdTest", outHost.data(), {outHost.size()},
                        outModel.data(), outModel.size(), 0.05, 0.05);
  }

  return true;
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  std::string operation;
  bool doReport = false;
  bool doPrintTensors = false;
  bool inPlace = true;
  bool ignoreData = false;
  bool enableOptimisations = true;
  unsigned tiles = 1;
  unsigned regionsPerTile = 1;
  ShapeOption<unsigned> pattern;
  po::options_description desc("Options");

  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
    ("print",
     po::value<bool>(&doPrintTensors)->default_value(doPrintTensors),
     "Print the tensors")
    ("ignore-data",
     po::value<bool>(&ignoreData)->default_value(ignoreData),
     "Ignore values of data, useful for benchmarking without "
     "overhead of upload/download of tensors")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type")
    ("in-place",
     po::value<bool>(&inPlace)->default_value(inPlace),
     "Do the specified operation in place")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("pattern",
     po::value<ShapeOption<unsigned>>(&pattern)->multitoken(),
     "Pattern as (innerFactor, motifLength, outerFactor)")
    ("regions-per-tile",
     po::value<unsigned>(&regionsPerTile)->default_value(regionsPerTile),
     "Number of patterned regions per tile")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Allowed operations: ADD MULTIPLY SUBTRACT\n")
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcasted vector op optimisations");
  // clang-format on

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
  if (operation == "ADD") {
    broadcastOperation = expr::BroadcastOpType::ADD;
    broadcastHostFn = [](double x, double y) -> double { return x + y; };
  } else if ((operation == "MULTIPLY") || (operation == "MUL")) {
    broadcastOperation = expr::BroadcastOpType::MULTIPLY;
    broadcastHostFn = [](double x, double y) -> double { return x * y; };
  } else if ((operation == "SUBTRACT") || (operation == "SUB")) {
    broadcastOperation = expr::BroadcastOpType::SUBTRACT;
    broadcastHostFn = [](double x, double y) -> double { return x - y; };
  } else {
    std::cerr << " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  // Check that the pattern has been specific correctly
  if (pattern.val.size() != 3) {
    std::cerr << " Error: " << pattern.val.size()
              << " integers supplied for pattern"
                 " specifier (exactly 3 required)! \n";
    return 1;
  }

  if (tiles < 1) {
    std::cerr << "Error: At least 1 tiles required!\n";
    return 1;
  }

  // Multiple tiles are required by the tool implementation in order create
  // multiple patterned regions.
  if ((regionsPerTile > 1) && (tiles < 2)) {
    std::cerr << "Error: Multiple regions are constructed by slicing a tensor"
                 " into multiple tiles. Hence at least two tiles are required "
                 "in order to"
                 " create multiple regions!\n";
    return 1;
  }

  // Report error for degenerate cases
  if ((pattern.val[1] == 1) && (pattern.val[2] > 1)) {
    std::cerr << "Error: This pattern cannot be created, as it is equivalent to"
                 " the pattern ("
              << (pattern.val[0] * pattern.val[2]) << ", 1, 1)!\n";
    return 1;
  }

  return !doBroadcastVectorOptimiseTest(
      deviceType, tiles, dataType, pattern.val[0], pattern.val[1],
      pattern.val[2], regionsPerTile, broadcastOperation, inPlace,
      broadcastHostFn, doReport, doPrintTensors, ignoreData,
      enableOptimisations);
}
