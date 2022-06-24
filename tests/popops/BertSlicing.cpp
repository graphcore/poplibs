// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
// Tests built to exercise sequenceSlice() in a simplification of the bert
// attention loop.

#define BOOST_TEST_MODULE BertSlicing

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <math.h>
#include <random>

#include <poplar/Engine.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/SequenceSlice.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VarStructure.hpp>

// Reference data generated from previous execution with genData=true
#include "popops/BertSlicingRefSmall.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;

// Check slicing in a similar manner to the bert attention layer.
// Some stages are not included, and the tile mapings are not representative of
// what frameworks will achieve.
BOOST_AUTO_TEST_CASE(AttentionSlice) {
  bool verbose = false;
  // enable to output data to be copied to BertSlicingRefSmall.hpp
  bool genData = false;
  if (isSimulator(TEST_TARGET)) {
    std::cout << "AttentionSlice is too slow to check on simulators\n";
    return;
  }
  auto device = createTestDeviceFullSize(TEST_TARGET);
  Target target = device.getTarget();

  std::cout << "Creating graph\n";
  Graph graph(device.getTarget());
  Sequence prog;
  DebugContext di{"SimpleSliceTest"};
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  poplin::PlanningCache planningCache;
  constexpr unsigned maxTokensPerBatch = 4 * 512;
  constexpr unsigned maxSequencesPerBatch = 8;
  constexpr unsigned maxTokensPerSequence = 512;
  constexpr unsigned nSequencesPerInnerLoop = 2;

  constexpr bool large = true;
  // BERT large causes problems with a large variable not fitting on the
  // cpu/ipuModel targets. So we run the large test to check for size and
  // the smaller tests to check for correctness.
  const bool manyFeatures = large && isHw(TEST_TARGET);
  const unsigned nFeatures = manyFeatures ? 1024 : 64;
  constexpr unsigned nHidden = large ? 64 : 8;
  constexpr unsigned nHeads = large ? 16 : 8;

  Tensor tOffsets = graph.addVariable(UNSIGNED_INT, {maxSequencesPerBatch},
                                      {di, "seqOffsets"});
  Tensor tLengths = graph.addVariable(UNSIGNED_INT, {maxSequencesPerBatch},
                                      {di, "seqLengths"});
  graph.setTileMapping(tOffsets, 0);
  graph.setTileMapping(tLengths, 0);

  // Incoming symbols.
  // TBD: map these more realistically.
  Tensor tBatchIn = graph.addVariable(HALF, {maxTokensPerBatch, nFeatures},
                                      {di, "batchTokensIn"});
  poputil::mapTensorLinearly(graph, tBatchIn);

  boost::multi_array<double, 2> hQkvWeights(
      boost::extents[nFeatures][3 * nHeads * nHidden]);
  Tensor qkvWeights = poplin::createMatMulInputRHS(
      graph, HALF, HALF, {maxTokensPerBatch, nFeatures},
      {nFeatures, 3 * nHeads * nHidden}, {di, "QKVWeights"});

  // Project. qkv={maxTokensPerBatch, nHeads * 3 * nHidden}
  Tensor qkv = poplin::matMul(graph, tBatchIn, qkvWeights, prog, HALF,
                              {di, "input projection qkv=batchIn*qkvWeights"});
  auto qkvGroupings = poputil::detectDimGroupings(graph, qkv);
  std::cerr << "qkv shape " << qkv.shapeToString() << ", expr " << qkv << "\n";
  for (const auto &grouping : qkvGroupings)
    std::cerr << "qkv grouping " << grouping.first << " : " << grouping.second
              << "\n";

  // Ignore masking.
  qkv = qkv.reshape({maxTokensPerBatch, nHeads, 3, nHidden});
  Tensor q = qkv.slice({0, 0, 0, 0}, {maxTokensPerBatch, nHeads, 1, nHidden});
  Tensor k = qkv.slice({0, 0, 1, 0}, {maxTokensPerBatch, nHeads, 2, nHidden});
  Tensor v = qkv.slice({0, 0, 2, 0}, {maxTokensPerBatch, nHeads, 3, nHidden});

  // Apply attention on nSequencesPerInnerLoop at a time.
  // Loop counter.
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {di, "seqIdx"});
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, {di, "one"});
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, prog, {di, "initSeqIdx"});
  Sequence loop;
  DebugContext loopDi{di, "Attention Inner Loop"};
  // Calculate slicing indices and lengths.
  // Note this is taking a single segment per seq, it needs to be 2 to support
  // masking.
  std::vector<unsigned> innerOffsets;
  for (unsigned i{0}; i != nSequencesPerInnerLoop; ++i)
    innerOffsets.emplace_back(i * maxTokensPerSequence);

  if (verbose)
    std::cerr << "tOffsets shape " << tOffsets.shapeToString() << "\n";

  Tensor tBatchOffsets =
      popops::dynamicSlice(
          graph,
          tOffsets.reshape({maxSequencesPerBatch / nSequencesPerInnerLoop,
                            nSequencesPerInnerLoop}),
          seqIdx, {0}, {1}, loop, {loopDi, "sliceOffsets"})
          .flatten();
  Tensor tInnerLengths =
      popops::dynamicSlice(
          graph,
          tLengths.reshape({maxSequencesPerBatch / nSequencesPerInnerLoop,
                            nSequencesPerInnerLoop}),
          seqIdx, {0}, {1}, loop, {loopDi, "sliceLengths"})
          .flatten();

  // Slice inputs.
  Tensor matMulQInput = poplin::createMatMulGroupedInputLHS(
      graph, HALF, HALF,
      {nHeads * nSequencesPerInnerLoop, maxTokensPerSequence, nHidden},
      {nHeads * nSequencesPerInnerLoop, nHidden, maxTokensPerSequence},
      {loopDi, "matMulQInput"}, {}, &planningCache);
  Tensor matMulKTInput = poplin::createMatMulGroupedInputRHS(
      graph, HALF, HALF,
      {nHeads * nSequencesPerInnerLoop, maxTokensPerSequence, nHidden},
      {nHeads * nSequencesPerInnerLoop, nHidden, maxTokensPerSequence},
      {loopDi, "matMulKInput"}, {}, &planningCache);
  Tensor matMulVInput = poplin::createMatMulGroupedInputLHS(
      graph, HALF, HALF,
      {nHeads * nSequencesPerInnerLoop, nHidden, maxTokensPerSequence},
      {nHeads * nSequencesPerInnerLoop, maxTokensPerSequence,
       maxTokensPerSequence},
      {loopDi, "matMulVInput"}, {}, &planningCache);
  Tensor tInnerOffsets =
      graph.addConstant(UNSIGNED_INT, {nSequencesPerInnerLoop},
                        innerOffsets.data(), {loopDi, "inner offsets"});
  graph.setTileMapping(tInnerOffsets, 0);
  // Rearrange the matmul inputs to make the nSequnecesPerInnerLoop and
  // tokensPerInnerLoop dimensions the outermost, then flatten them together.
  auto matMulQInputRearranged = matMulQInput
                                    .reshape({nSequencesPerInnerLoop, nHeads,
                                              maxTokensPerSequence, nHidden})
                                    .dimShuffle({0, 2, 1, 3})
                                    .flatten(0, 2);
  auto matMulKTInputRearranged = matMulKTInput
                                     .reshape({nSequencesPerInnerLoop, nHeads,
                                               nHidden, maxTokensPerSequence})
                                     .dimShuffle({0, 3, 1, 2})
                                     .flatten(0, 2);
  auto matMulVInputRearranged = matMulVInput
                                    .reshape({nSequencesPerInnerLoop, nHeads,
                                              nHidden, maxTokensPerSequence})
                                    .dimShuffle({0, 3, 1, 2})
                                    .flatten(0, 2);
  if (verbose) {
    std::cerr << "matMulQInput shape " << matMulQInput.shapeToString() << "\n";
    std::cerr << "q shape " << q.shapeToString() << "\n";
    std::cerr << "tInnerLengths shape " << tInnerLengths.shapeToString()
              << "\n";
    std::cerr << "tBatchOffsets shape " << tBatchOffsets.shapeToString()
              << "\n";
    std::cerr << "tInnerOffsets shape " << tInnerOffsets.shapeToString()
              << "\n";
    std::cerr << "k shape " << k.shapeToString() << "\n";
  }
  popops::sequenceSlice(graph, q, matMulQInputRearranged, tInnerLengths,
                        tBatchOffsets, tInnerOffsets, true, loop,
                        {loopDi, "getQ"});

  popops::sequenceSlice(graph, k, matMulKTInputRearranged, tInnerLengths,
                        tBatchOffsets, tInnerOffsets, true, loop,
                        {loopDi, "getK"});
  popops::sequenceSlice(graph, v, matMulVInputRearranged, tInnerLengths,
                        tBatchOffsets, tInnerOffsets, true, loop,
                        {loopDi, "getV"});

  // Grouped matmul to calculate q*kt {nHeads*nSequencesPerInnerLoop,
  // maxTokensPerSequence, maxTokensPerSequence}.
  Tensor qKt =
      poplin::matMulGrouped(graph, matMulQInput, matMulKTInput, loop, HALF,
                            {loopDi, "qKt=q*Kt"}, {}, &planningCache);
  // Ignore masking step.
  // Ignore softmax and scaling steps.
  // Grouped matmul to calculate z=(q*kt)*v
  //   {nSequencesPerInnerLoop*nHeads, nHidden, maxTokensPerSequence}
  Tensor z = poplin::matMulGrouped(graph, matMulVInput, qKt, loop, HALF,
                                   {loopDi, "z=v*qKt"}, {}, &planningCache);
  z = z.reshape(
      {nSequencesPerInnerLoop, nHeads, nHidden, maxTokensPerSequence});
  //{nSequencesPerInnerLoop * maxTokensPerSequence, nHeads, nHidden}
  z = z.dimShuffle({0, 3, 1, 2}).flatten(0, 2);

  // Add z into zBatch
  Tensor zBatch = poplin::createMatMulInputLHS(
      graph, HALF, HALF, {maxTokensPerBatch, nHeads * nHidden},
      {nHeads * nHidden, nFeatures}, {di, "zBatch"}, {}, &planningCache);

  popops::fill(graph, zBatch, prog, 0u, {di, "zero output batch"});
  if (verbose) {
    std::cerr << "z shape " << z.shapeToString() << "\n";
    std::cerr << "zBatch shape " << zBatch.shapeToString() << "\n";
  }
  popops::sequenceSlice(graph, z, zBatch, tInnerLengths, tInnerOffsets,
                        tBatchOffsets, false, loop, {loopDi, "updateOutput"});
  popops::addInPlace(graph, seqIdx, one, loop, {loopDi, "+1"});
  prog.add(Repeat(maxSequencesPerBatch / nSequencesPerInnerLoop, loop,
                  {loopDi, "loop"}));

  // Final projection matmul on the batch.
  boost::multi_array<double, 2> hWProjWeights(
      boost::extents[nHeads * nHidden][nFeatures]);

  Tensor wProj = poplin::createMatMulInputRHS(
      graph, HALF, HALF, {maxTokensPerBatch, nHeads * nHidden},
      {nHeads * nHidden, nFeatures}, {di, "WProjection"}, {}, &planningCache);
  if (verbose) {
    std::cerr << "zbatch" << zBatch.shapeToString() << "\n";
    std::cerr << "wProj" << wProj.shapeToString() << "\n";
  }
  Tensor tBatchOut =
      poplin::matMul(graph, zBatch, wProj, prog, HALF, {di, "output=z*wProj"});
  if (verbose) {
    std::cerr << "tBatchOut" << tBatchOut.shapeToString() << "\n";
  }

  graph.createHostWrite("offset", tOffsets);
  graph.createHostWrite("length", tLengths);
  graph.createHostWrite("qkvWeights", qkvWeights);
  graph.createHostWrite("wProj", wProj);
  graph.createHostWrite("tokensToIpu", tBatchIn);
  graph.createHostRead("tokensFromIpu", tBatchOut);

  std::cerr << "Create engine\n";
  Engine engine(graph, prog);
  // Check the model runs ok.
  std::mt19937 randomEngine(42);

  // Correctness of the model is not checked.
  device.bind([&](const Device &d) {
    std::cerr << "Load engine\n";
    engine.load(d);
    std::cerr << "Load params\n";
    // The tokens for each sequence must be non-overlapping and respect
    // maxTokensPerSequence
    std::vector<unsigned> hOffset{0, 16, 48, 176, 432, 944, 1200, 1712};
    std::vector<unsigned> hLength{16, 32, 128, 256, 512, 256, 512, 32};
    // Gap of 32 unused values at the end
    assert(hOffset.size() == hLength.size());
    for (std::size_t i{0}; i != hLength.size(); ++i) {
      // Check specificed sequences fit inside the token buffer
      BOOST_CHECK(hLength[i] <= maxTokensPerSequence);
      BOOST_CHECK(hOffset[i] + hLength[i] <= maxTokensPerBatch);
      // Check no overlap between sequences.
      if (i > 0)
        BOOST_CHECK(hOffset[i - 1] + hLength[i - 1] <= hOffset[i]);
    }
    boost::multi_array<double, 2> hTokens(
        boost::extents[maxTokensPerBatch][nFeatures]);
    auto &target = device.getTarget();
    writeRandomValues(target, HALF, hQkvWeights, -.09, +.1, randomEngine);
    writeRandomValues(target, HALF, hWProjWeights, -.09, +.1, randomEngine);
    writeRandomValues(target, HALF, hTokens, -1., +1., randomEngine);

    std::vector<char> rawQkvWeights(hQkvWeights.num_elements() *
                                    target.getTypeSize(HALF));
    copyDoubleToDeviceHalf(target, hQkvWeights.data(), rawQkvWeights.data(),
                           hQkvWeights.num_elements());
    std::vector<char> rawWProjWeights(hWProjWeights.num_elements() *
                                      target.getTypeSize(HALF));
    copyDoubleToDeviceHalf(target, hWProjWeights.data(), rawWProjWeights.data(),
                           hWProjWeights.num_elements());

    std::vector<char> rawTokens(hTokens.num_elements() *
                                target.getTypeSize(HALF));
    copyDoubleToDeviceHalf(target, hTokens.data(), rawTokens.data(),
                           hTokens.num_elements());

    engine.writeTensor("qkvWeights", ArrayRef(rawQkvWeights));
    engine.writeTensor("wProj", ArrayRef(rawWProjWeights));
    // engine.writeTensor("offset", ArrayRef(hOffset));
    engine.writeTensor("offset", hOffset.data(),
                       hOffset.data() + hOffset.size());
    engine.writeTensor("length", ArrayRef(hLength));
    engine.writeTensor("tokensToIpu", ArrayRef(rawTokens));
    engine.run();

    engine.readTensor("tokensFromIpu", rawTokens.data(),
                      rawTokens.data() + rawTokens.size());
    boost::multi_array<float, 2> hTokensOut(
        boost::extents[maxTokensPerBatch][nFeatures]);
    copyDeviceHalfToFloat(target, rawTokens.data(), hTokensOut.data(),
                          hTokensOut.num_elements());

    {
      unsigned n = 0;
      for (unsigned t = 0; t != hTokensOut.shape()[0]; ++t) {
        for (unsigned f = 0; f != hTokensOut.shape()[1]; ++f) {
          if (hTokensOut[t][f] != 0)
            ++n;
        }
      }
      std::cout << "Found " << n << "/" << hTokensOut.num_elements()
                << " nonzero elements\n";
      BOOST_CHECK(n > 0);
    }
    if (genData && !manyFeatures) {
      // Output the reference data is a form ready for clipping and pasting
      // into the reference data files.
      std::string name;
      name = manyFeatures ? "TokensLarge" : "TokensSmall";
      std::cout << "// Generated by recompiling with `genData` true.\n";
      std::cout << "std::vector<float> flat" << name << " {\n";
      std::cout << "//";
      for (unsigned t = 0; t != hTokensOut.shape()[0]; ++t) {
        std::cout << "{ " << t << "\n";
        for (unsigned f = 0; f != hTokensOut.shape()[1]; ++f) {
          std::cout << hTokensOut[t][f] << ",\n";
        }
        std::cout << "//},";
      }
      std::cout << "\n};\n";
      std::cout << "boost::multi_array_ref<float, 2> ref" << name << "(flat"
                << name << ".data(), boost::extents[" << hTokensOut.shape()[0]
                << "][" << hTokensOut.shape()[1] << "]);\n";
    }
    if (!manyFeatures)
      BOOST_CHECK(
          checkIsClose("outTokens", hTokensOut, refTokensSmall, .01, 1e-7));
    // When manyFeatures==true we're just checking that the model fitted and
    // ran OK.
  });
}
