// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

// Test for the Cast vertex
//
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/program_options.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

//*************************************************
// Test function for Cast
//
// Overview:
//
// Run a series of tests that cast the specified number of items.
// The results are put into a larger memory area and the remaining items are
// expected to be zero.  This is checked as well as the "wanted" data.
//*************************************************
bool doTest(const DeviceType &deviceType, Type &dataTypeIn, Type &dataTypeOut,
            unsigned rows, unsigned columns, unsigned offsetOut,
            const bool supervisor, const Fp8Format fp8Format,
            const int fp8Scale, const Fp8Format fp8FormatOut,
            const int fp8ScaleOut) {

  // Check that the output offset results in a multiple of 4
  // bytes
  if (offsetOut % 2 != 0) {
    throw std::logic_error("Offset is not a multiple of output alignment, "
                           "copies will be introduced");
  }

  // Whole data array size
  auto total_elems = rows * columns;
  auto total_size = rows * (columns + offsetOut);

  // Program generated test data
  std::vector<double> outTest(total_size);
  std::vector<double> inTest(total_elems);

  // Initialise input pattern, picking a numeric range and
  // tolerance (below) that works for halves as a limited size/resolution data
  // type with enough unique numbers to satisfy a large test size
  if (dataTypeIn == QUARTER || dataTypeOut == QUARTER) {
    // Pick values that are exact for the FP8 format selected.
    // QUART 143 has 3 mantissa bits + 1 lead bit = 4 bits, supports 0..16
    // QUART 152 has 2 mantissa bits + 1 lead bit = 3 bits, supports 0..8
    const unsigned modulo = fp8Format == Fp8Format::QUART143 ? 17 : 9;
    for (unsigned i = 0; i < total_elems; i++) {
      inTest[i] = (i + 1) % modulo;
      if (i % 2 && dataTypeIn == CHAR) {
        inTest[i] *= -1;
      }
    }
  } else if (dataTypeIn == CHAR || dataTypeIn == SIGNED_CHAR) {
    for (unsigned i = 0; i < total_elems; i++)
      inTest[i] = static_cast<signed char>(i);
  } else {
    for (unsigned i = 0; i < total_elems; i++)
      inTest[i] = 0.1 * i + 1;
  }

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Input data
  Tensor in = graph.addVariable(dataTypeIn, {rows, columns}, "Input Data");
  graph.setTileMapping(in, 0);

  // Intermediate data - we can check the following without extra support
  // half to fp8 to half
  // char to fp8 to char
  // fp8 to fp8 to fp8 (Verified as char values, and changing fp8 type)
  bool fp8 = dataTypeIn == QUARTER || dataTypeOut == QUARTER;
  bool fp8ToFp8 = dataTypeIn == QUARTER && dataTypeOut == QUARTER;
  bool inTypeToFp8ToinType = fp8 && dataTypeOut == QUARTER && !fp8ToFp8;

  if (dataTypeIn == QUARTER) {
    std::cout << "WARNING: Codelets will be run for quarter->" << dataTypeOut
              << " but result checking is incomplete\n";
  }

  Tensor inter;
  if (inTypeToFp8ToinType) {
    std::cout << "Using the process " << dataTypeIn << "->quarter->"
              << dataTypeIn
              << " to check casting to/from "
                 "fp8 data\n";
    inter =
        graph.addVariable(dataTypeOut, {rows, columns}, "Intermediate Data");
    graph.setTileMapping(inter, 0);
  }

  // Result data
  auto resultDataType = inTypeToFp8ToinType ? dataTypeIn : dataTypeOut;
  Tensor out =
      graph.addVariable(resultDataType, {rows, columns + offsetOut}, "Output");
  graph.setTileMapping(out, 0);

  // allocateHostMemoryForTensor
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto input = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                           downloadProg, tmap);

  auto output = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  // Make a sequence to zero output memory and run cast
  Sequence sequence;

  ComputeSet testComputeSet = graph.addComputeSet("computeCast");

  std::string vertexName;
  if (supervisor) {
    vertexName = "popops::Cast1D";
  } else if (rows == 1) {
    vertexName = "popops::Cast1DSingleWorker";
  } else {
    vertexName = "popops::Cast2D";
  }

  if (in.elementType() == QUARTER) {
    graph.setInitialValue(in.getMetadata(),
                          packFp8MetaData(fp8Format, fp8Scale));
  }

  if (inTypeToFp8ToinType && inter.elementType() == QUARTER) {
    graph.setInitialValue(inter.getMetadata(),
                          packFp8MetaData(fp8Format, fp8Scale));
  }

  if (out.elementType() == QUARTER) {
    if (fp8ToFp8) {
      graph.setInitialValue(out.getMetadata(),
                            packFp8MetaData(fp8FormatOut, fp8ScaleOut));
    } else {
      graph.setInitialValue(out.getMetadata(),
                            packFp8MetaData(fp8Format, fp8Scale));
    }
  }

  auto castVertex = graph.addVertex(
      testComputeSet, templateVertex(vertexName, dataTypeIn, dataTypeOut));
  graph.setTileMapping(castVertex, 0);

  // Use slices to apply the offset, and deal with 1d/ 2d cases
  Tensor sliceIn, sliceInter, sliceOut;
  if (rows > 1) {
    sliceIn = in.slice({0, 0}, {rows, columns});
    sliceOut = out.slice({0, offsetOut}, {rows, columns + offsetOut});
    if (inTypeToFp8ToinType) {
      sliceInter = inter.slice({0, 0}, {rows, columns});
    }
  } else {
    sliceIn = in.reshape({columns});
    sliceOut = out.reshape({columns + offsetOut});
    sliceOut = sliceOut.slice(offsetOut, columns + offsetOut);
    if (inTypeToFp8ToinType) {
      sliceInter = inter.reshape({columns});
    }
    unsigned totElems = sliceIn.numElements();
    graph.setInitialValue(castVertex["numElems"], totElems);
  }

  graph.connect(castVertex["src"], sliceIn);
  if (inTypeToFp8ToinType) {
    graph.connect(castVertex["dst"], sliceInter);
  } else {
    graph.connect(castVertex["dst"], sliceOut);
  }

  popops::zero(graph, out, sequence, "Zero output");
  sequence.add(Execute(testComputeSet));
  if (inTypeToFp8ToinType) {
    ComputeSet testComputeSet = graph.addComputeSet("computeCastInterToOut");
    auto castVertex = graph.addVertex(
        testComputeSet, templateVertex(vertexName, dataTypeOut, dataTypeIn));
    graph.setTileMapping(castVertex, 0);
    graph.connect(castVertex["src"], sliceInter);
    graph.connect(castVertex["dst"], sliceOut);
    if (rows == 1) {
      unsigned totElems = sliceInter.numElements();
      graph.setInitialValue(castVertex["numElems"], totElems);
    }

    sequence.add(Execute(testComputeSet));
  }

  // Run each sequence and compare host and IPU result
  Engine engine(graph, Sequence{uploadProg, sequence, downloadProg});
  attachStreams(engine, tmap);

  // Put test inputs into an array of the correct type ready to use
  copy(target, inTest.data(), inTest.size(),
       fp8ToFp8 ? UNSIGNED_CHAR : dataTypeIn, input.get());

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  std::vector<double> outHost(total_size);
  copy(target, fp8ToFp8 ? UNSIGNED_CHAR : resultDataType, output.get(),
       outHost.data(), outHost.size());

  // Host generated result, start with zeros
  for (unsigned i = 0; i < total_size; i++)
    outTest[i] = 0;
  // Then cast the same portion of the input as the code under test
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < columns; j++) {
      outTest[j + i * (columns + offsetOut) + offsetOut] =
          inTest[j + i * columns];
    }
  }
  // Check the result, in the outTest array
  // Always check the whole output memory to catch any overwrites

  bool check = checkIsClose("CastTest", outHost.data(), {outHost.size()},
                            outTest.data(), outTest.size(), 0.05, 0.05);

  return check;
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type inType;
  Type outType;
  unsigned rows, columns, offsetOut;
  bool supervisor = false;
  Fp8Format fp8Format = Fp8Format::QUART143;
  Fp8Format fp8FormatOut = Fp8Format::QUART143;
  int fp8Scale = 0, fp8ScaleOut = 0;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("in-type",
     po::value<Type>(&inType)->required(),
     "Input Type")
    ("out-type",
     po::value<Type>(&outType)->required(),
     "Output Type")
    ("fp8-scale",
     po::value<int>(&fp8Scale)->implicit_value(fp8Scale),
     "Exponent scale for fp8 type conversion")
    ("fp8-format",
     po::value<Fp8Format>(&fp8Format)->implicit_value(fp8Format),
     "Format for fp8 type conversion")
    ("fp8-scale-out",
     po::value<int>(&fp8ScaleOut)->implicit_value(fp8ScaleOut),
     "Output exponent scale for fp8 type conversion when casting fp8->fp8")
    ("fp8-format-out",
     po::value<Fp8Format>(&fp8FormatOut)->implicit_value(fp8FormatOut),
     "Output format for fp8 type conversion when casting fp8->fp8")
    ("rows",
     po::value<unsigned>(&rows)->required(),
     "In/Out data rows")
    ("columns",
     po::value<unsigned>(&columns)->required(),
     "In/Out data columns")
    ("out-offset",
     po::value<unsigned>(&offsetOut)->required(),
     "Output offset in output word size units")
    ("supervisor",
     po::value<bool>(&supervisor)->implicit_value(true),
     "Use supervisor vertex (only valid if rows=1)");
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

  if (supervisor && (rows > 1)) {
    std::cerr << "error: 'supervisor' option requires 'rows'=1\n";
    return 1;
  }
  if (!doTest(deviceType, inType, outType, rows, columns, offsetOut, supervisor,
              fp8Format, fp8Scale, fp8FormatOut, fp8ScaleOut))
    return 1;
  return 0;
}
