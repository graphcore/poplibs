// Copyright (c) Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE BlockSparseOpsTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/codelets.hpp>
#include <popops/EncodingConstants.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <random>
#include <unordered_map>
#include <vector>

#include "popsparse/BSMatrix.hpp"
#include "popsparse/BSNonLinearity.hpp"
#include "popsparse/BSOps.hpp"
#include "popsparse/experimental/BlockSparse.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popsparse::experimental;
using namespace poputil;
using namespace popnn;

unsigned computeNz(unsigned blockRows, unsigned blockCols,
                   const unsigned char *sparsity,
                   std::vector<unsigned> &nzBlocksByRow,
                   std::vector<unsigned> &nzBlocksByRowTotal) {
  unsigned nz = 0;
  unsigned total = 0;
  nzBlocksByRow.resize(blockRows, 0);
  nzBlocksByRowTotal.resize(blockRows, 0);
  for (unsigned int br = 0; br < blockRows; ++br) {
    for (unsigned int bc = 0; bc < blockCols; ++bc) {
      unsigned idxBlockDense = br * blockCols + bc;
      if (sparsity[idxBlockDense]) {
        ++nzBlocksByRow[br];
        ++nz;
      }
    }
    nzBlocksByRowTotal[br] = total;
    total += nzBlocksByRow[br];
  }
  return nz;
}

#define USE_RANDOM_VALUES 1

void populateSparseBlocks(unsigned blockRow, unsigned blockCol,
                          unsigned blockRows, unsigned blockCols, unsigned nz,
                          const std::vector<unsigned> &nzBlocksByRow,
                          bool needTranspose,
                          std::vector<float> &valuesRowMjSparse,
                          std::vector<float> &valuesBlockSparse) {
  const unsigned blockArea = blockRow * blockCol;
#if USE_RANDOM_VALUES
#if 0
  // Random testing. Avoid in production
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 randomEngine(seed);
#endif
  std::mt19937 randomEngine;
#endif
  valuesRowMjSparse.resize(nz * blockArea);
  valuesBlockSparse.resize(valuesRowMjSparse.size());

#if USE_RANDOM_VALUES
  for (unsigned i = 0; i < valuesRowMjSparse.size(); ++i) {
    valuesRowMjSparse[i] = static_cast<float>(randomEngine()) /
                           static_cast<float>(randomEngine.max());
  }
#endif
  for (unsigned br = 0, idxBlock = 0, idxBlockComplete = 0; br < blockRows;
       ++br) {
    for (unsigned bcPacked = 0; bcPacked < nzBlocksByRow[br];
         ++bcPacked, ++idxBlock) {
      for (unsigned rb = 0; rb < blockRow; ++rb) {
        for (unsigned cb = 0; cb < blockCol; ++cb) {
          unsigned idxInBlock =
              !needTranspose ? (rb * blockCol + cb) : (cb * blockRow + rb);
          unsigned idxDense = idxBlockComplete * blockArea +
                              rb * nzBlocksByRow[br] * blockCol +
                              bcPacked * blockCol + cb;
#if !USE_RANDOM_VALUES
          valuesRowMjSparse[idxDense] = idxBlock;
#endif
          valuesBlockSparse[idxBlock * blockArea + idxInBlock] =
              valuesRowMjSparse[idxDense];
        }
      }
    }
    idxBlockComplete += nzBlocksByRow[br];
  }
}

/*
Testing slice()

Block: 2x2

Dense block shape:
1 . 2
. 3 4
5 6 .

Sparse block shape:
1 2 3 4 5 6

*/
BOOST_AUTO_TEST_CASE(slice_test) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const unsigned blockRow = 2;
  const unsigned blockCol = 2;
  const unsigned blockRows = 3;
  const unsigned blockCols = 3;
  const unsigned blockArea = blockRow * blockCol;

  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[0 * blockCols + 2] = 1; // 0,2
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[1 * blockCols + 2] = 1; // 1,2
  sparsity[2 * blockCols + 0] = 1; // 2,0
  sparsity[2 * blockCols + 1] = 1; // 2,1

  std::vector<unsigned> nzBlocksByRow;
  std::vector<unsigned> nzBlocksByRowTotal;
  unsigned nz = computeNz(blockRows, blockCols, sparsity.data(), nzBlocksByRow,
                          nzBlocksByRowTotal);
  std::vector<float> valuesRowMjSparse;
  std::vector<float> valuesBlockSparse;
  populateSparseBlocks(blockRow, blockCol, blockRows, blockCols, nz,
                       nzBlocksByRow, false, valuesRowMjSparse,
                       valuesBlockSparse);

  Tensor sparseTensor = graph.addConstant(poplar::FLOAT, {nz, blockArea},
                                          valuesBlockSparse.data(), "sparse");

  BlockSparseMatrix bm(blockRow * blockRows, blockCol * blockCols, blockRow,
                       blockCol, false, sparsity.data());
  Tensor sparseTensorShapeCheck =
      bm.createTensor(graph, FLOAT, "sparseShapeCheck");
  BOOST_TEST(sparseTensor.shape() == sparseTensorShapeCheck.shape());

  //      coord dim
  // slice(1,    0) => [3,4]
  unsigned rowSlice = 1;
  Tensor slice0 = slice(sparseTensor, rowSlice, 0, blockRow, blockCol,
                        blockRows, blockCols, false, sparsity.data());

  BOOST_TEST(slice0.shape() == std::vector<std::size_t>({2 * blockCol}));

  const float epsilon = 0.001f;
  float valueRead;
  unsigned brSlice = rowSlice / blockRow;
  unsigned rbSlice = rowSlice % blockRow;
  for (unsigned bc = 0, colDense = 0; bc < blockCols; ++bc) {
    unsigned idxDense = brSlice * blockCols + bc;
    if (sparsity[idxDense]) {
      for (unsigned cb = 0; cb < blockCol; ++cb, ++colDense) {
        bool ret = slice0[colDense].getConstantValue(&valueRead);
        BOOST_TEST(ret);
        unsigned idxDense = nzBlocksByRowTotal[brSlice] * blockArea +
                            rbSlice * nzBlocksByRow[brSlice] * blockCol +
                            colDense;
        float valueExpected = valuesRowMjSparse[idxDense];
        float err = fabs(valueRead - valueExpected);
        if (err > epsilon) {
          BOOST_TEST(err <= epsilon);
        }
      }
    }
  }

  //      coord dim
  // slice(0,    1) => [1,5]
  unsigned colSlice = 0;
  Tensor slice1 = slice(sparseTensor, colSlice, 1, blockRow, blockCol,
                        blockRows, blockCols, false, sparsity.data());

  BOOST_TEST(slice1.shape() == std::vector<std::size_t>({2 * blockRow}));
  unsigned bcSlice = colSlice / blockCol;
  for (unsigned br = 0, rowDense = 0; br < blockRows; ++br) {
    unsigned idxDense = br * blockCols + bcSlice;
    if (sparsity[idxDense]) {
      for (unsigned rb = 0; rb < blockRow; ++rb, ++rowDense) {
        bool ret = slice1[rowDense].getConstantValue(&valueRead);
        BOOST_TEST(ret);
        unsigned idxDense = nzBlocksByRowTotal[br] * blockArea +
                            rb * nzBlocksByRow[br] * blockCol + colSlice;
        float valueExpected = valuesRowMjSparse[idxDense];
        float err = fabs(valueRead - valueExpected);
        if (err > epsilon) {
          BOOST_TEST(err <= epsilon);
        }
      }
    }
  }
}

void softmaxTest(unsigned blockRow, unsigned blockCol, unsigned blockRows,
                 unsigned blockCols, const std::vector<unsigned char> &sparsity,
                 const Type &dataType, bool filterUpperTriangle, bool inPlace) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  const unsigned blockArea = blockRow * blockCol;
  const unsigned rows = blockRows * blockRow;

  std::vector<unsigned> nzBlocksByRow;
  std::vector<unsigned> nzBlocksByRowTotal;
  unsigned nz = computeNz(blockRows, blockCols, sparsity.data(), nzBlocksByRow,
                          nzBlocksByRowTotal);
  std::vector<float> valuesRowMjSparse;
  std::vector<float> valuesBlockSparse;
  populateSparseBlocks(blockRow, blockCol, blockRows, blockCols, nz,
                       nzBlocksByRow, false, valuesRowMjSparse,
                       valuesBlockSparse);

  std::vector<std::vector<float>> valuesRowSparse(rows);
  for (unsigned br = 0, r = 0, idxElem = 0; br < blockRows; ++br) {
    for (unsigned rb = 0; rb < blockRow; ++rb, ++r) {
      for (unsigned bc = 0, c = 0; bc < blockCols; ++bc) {
        unsigned idxDense = br * blockCols + bc;
        for (unsigned cb = 0; cb < blockCol; ++cb, ++c) {
          if (sparsity[idxDense]) {
            if (!filterUpperTriangle || r >= c) {
              valuesRowSparse[r].push_back(valuesRowMjSparse[idxElem]);
            }
            ++idxElem;
          }
        }
      }
    }
  }

  std::vector<Tensor> packedDenseTensors(rows);
  for (unsigned r = 0; r < rows; ++r) {
    packedDenseTensors[r] = graph.addConstant(
        dataType, {1, valuesRowSparse[r].size()}, valuesRowSparse[r].data(),
        std::string("packedDense") + std::to_string(r));
    mapTensorLinearly(graph, packedDenseTensors[r]);
  }

  std::vector<std::size_t> sparseShape = {nz, blockArea};
  Tensor sparseTensorC = graph.addConstant(dataType, sparseShape,
                                           valuesBlockSparse.data(), "sparseC");
  mapTensorLinearly(graph, sparseTensorC);

  Tensor sparseTensor = graph.addVariable(dataType, sparseShape, "sparse");
  mapTensorLinearly(graph, sparseTensor);

  Sequence softmaxProg;
  softmaxProg.add(program::Copy(sparseTensorC, sparseTensor));

  Tensor bsSoftmax =
      bsSoftmaxInternal(graph, sparseTensor, inPlace, blockRow, blockCol,
                        blockRows, blockCols, sparsity.data(),
                        filterUpperTriangle ? SubBlockMask::ZeroUpperTriangle
                                            : SubBlockMask::None,
                        softmaxProg, "bSsoftmax");
  if (inPlace) {
    bsSoftmax = sparseTensor;
  }

  std::vector<Tensor> nnSoftmaxs(rows);
  for (unsigned int r = 0; r < rows; ++r) {
    nnSoftmaxs[r] =
        nonLinearity(graph, NonLinearityType::SOFTMAX, packedDenseTensors[r],
                     softmaxProg, std::string("nNsoftmax") + std::to_string(r));
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  std::unique_ptr<char[]> bsSoftmaxRawHost =
      poplibs_test::util::allocateHostMemoryForTensor(
          bsSoftmax, "bsSoftmax", graph, uploadProg, downloadProg, streamMaps);

  std::vector<std::unique_ptr<char[]>> nnSoftmaxRawHosts(rows);
  for (unsigned int r = 0; r < rows; ++r) {
    nnSoftmaxRawHosts[r] = poplibs_test::util::allocateHostMemoryForTensor(
        nnSoftmaxs[r], std::string("nnSoftmax") + std::to_string(r), graph,
        uploadProg, downloadProg, streamMaps);
  }

  Sequence mainSequence;
  mainSequence.add(uploadProg);
  mainSequence.add(softmaxProg);
  mainSequence.add(poplar::program::PrintTensor("bsSoftmax", bsSoftmax));
  for (unsigned int r = 0; r < rows; ++r) {
    mainSequence.add(poplar::program::PrintTensor(
        std::string("nnSoftmaxs") + std::to_string(r), nnSoftmaxs[r]));
  }
  mainSequence.add(downloadProg);

  const OptionFlags engineOptions{{"target.workerStackSizeInBytes", "0x200"},
                                  {"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, mainSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  boost::multi_array<float, 2> bsSoftmaxHost(
      boost::extents[sparseShape[0]][sparseShape[1]]);
  poplibs_test::util::copy(target, dataType, bsSoftmaxRawHost.get(),
                           bsSoftmaxHost);

  std::vector<boost::multi_array<float, 2>> nnSoftmaxHosts;
  for (unsigned int r = 0; r < rows; ++r) {
    boost::multi_array<float, 2> nnSoftmaxHost(
        boost::extents[1][valuesRowSparse[r].size()]);
    poplibs_test::util::copy(target, dataType, nnSoftmaxRawHosts[r].get(),
                             nnSoftmaxHost);
    nnSoftmaxHosts.push_back(nnSoftmaxHost);
  }

  const float epsilon = (dataType == FLOAT ? 0.001f : 0.1f);
  for (unsigned br = 0, idxBlock = 0; br < blockRows; ++br) {
    for (unsigned bcPacked = 0; bcPacked < nzBlocksByRow[br];
         ++bcPacked, ++idxBlock) {
      for (unsigned rb = 0; rb < blockRow; ++rb) {
        for (unsigned cb = 0; cb < blockCol; ++cb) {
          unsigned rowDense = br * blockRow + rb;
          unsigned colDense = bcPacked * blockCol + cb;
          unsigned idxInBlock = rb * blockCol + cb;
          // We compare the result only when softmaxrow is non-empty
          float valueBsSoftmax = bsSoftmaxHost[idxBlock][idxInBlock];
          float valueNnSoftmax;
          if (colDense < nnSoftmaxHosts[rowDense].shape()[1]) {
            valueNnSoftmax = nnSoftmaxHosts[rowDense][0][colDense];
          } else {
            valueNnSoftmax = 0.0f;
          }
          float err = fabs(valueBsSoftmax - valueNnSoftmax);
          if (err > epsilon) {
            BOOST_TEST(err <= epsilon);
          }
        }
      }
    }
  }
}

// Testing softmax()
/*
Block: 2x2

Dense block shape:
1 . .
2 3 .
. 4 .

Sparse block shape:
1 2 3 4

Dense block shapes to compare:
1
2 3
4

*/
BOOST_AUTO_TEST_CASE(softmax_testF32) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 2;
  const unsigned blockRows = 3;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[1 * blockCols + 0] = 1; // 1,0
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 1] = 1; // 2,1

  softmaxTest(blockRow, blockCol, blockRows, blockCols, sparsity, FLOAT, false,
              false);
}

BOOST_AUTO_TEST_CASE(softmax_testF32inPlace) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 2;
  const unsigned blockRows = 3;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[1 * blockCols + 0] = 1; // 1,0
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 1] = 1; // 2,1

  softmaxTest(blockRow, blockCol, blockRows, blockCols, sparsity, FLOAT, false,
              true);
}

BOOST_AUTO_TEST_CASE(softmax_testF16) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 2;
  const unsigned blockRows = 3;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[1 * blockCols + 0] = 1; // 1,0
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 1] = 1; // 2,1

  softmaxTest(blockRow, blockCol, blockRows, blockCols, sparsity, HALF, false,
              false);
}

// Testing softmax() with subblock mask
/*
Block: 2x3
Blocks: 4x3

Block mask:
1 . 1
. 1 .
1 . .
. . 1

Dense shape:
0 0 0 . . . . . .
0 0 0 . . . . . .
. . . 1 1 1 . . .
. . . 1 1 1 . . .
2 2 2 . . . . . .
2 2 2 . . . . . .
. . . . . . 3 3 3
. . . . . . 3 3 3
. . . . . . . . .
. . . . . . . . .

Dense masked shape:
0 . . . . . . . .
0 0 . . . . . . .
. . . . . . . . .
. . . 1 . . . . .
2 2 2 . . . . . .
2 2 2 . . . . . .
. . . . . . 3 . .
. . . . . . 3 3 .
. . . . . . . . .
. . . . . . . . .

*/
BOOST_AUTO_TEST_CASE(softmaxSubBlockMask_testF32) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 3;
  const unsigned blockRows = 5;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 0] = 1; // 2,0
  sparsity[3 * blockCols + 2] = 1; // 3,2

  softmaxTest(blockRow, blockCol, blockRows, blockCols, sparsity, FLOAT, true,
              false);
}

BOOST_AUTO_TEST_CASE(softmaxSubBlockMask_testF16) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 3;
  const unsigned blockRows = 4;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 0] = 1; // 2,0
  sparsity[3 * blockCols + 2] = 1; // 3,2

  softmaxTest(blockRow, blockCol, blockRows, blockCols, sparsity, HALF, true,
              false);
}

void softmaxGradTest(unsigned blockRow, unsigned blockCol, unsigned blockRows,
                     unsigned blockCols,
                     const std::vector<unsigned char> &sparsity,
                     const Type &outType, const Type &outGradType) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  const unsigned blockArea = blockRow * blockCol;
  const unsigned rows = blockRows * blockRow;

  std::vector<unsigned> nzBlocksByRow;
  std::vector<unsigned> nzBlocksByRowTotal;
  unsigned nz = computeNz(blockRows, blockCols, sparsity.data(), nzBlocksByRow,
                          nzBlocksByRowTotal);
  std::vector<float> vOutRowMjSparse;
  std::vector<float> vOutBlockSparse;
  populateSparseBlocks(blockRow, blockCol, blockRows, blockCols, nz,
                       nzBlocksByRow, false, vOutRowMjSparse, vOutBlockSparse);

  std::vector<float> vOutGradRowMjSparse;
  std::vector<float> vOutGradBlockSparse;
  populateSparseBlocks(blockRow, blockCol, blockRows, blockCols, nz,
                       nzBlocksByRow, false, vOutGradRowMjSparse,
                       vOutGradBlockSparse);

  std::vector<std::vector<float>> vOutRowSparse(rows);
  std::vector<std::vector<float>> vOutGradRowSparse(rows);
  for (unsigned br = 0, r = 0, idxElem = 0; br < blockRows; ++br) {
    for (unsigned rb = 0; rb < blockRow; ++rb, ++r) {
      for (unsigned bc = 0, c = 0; bc < blockCols; ++bc) {
        unsigned idxDense = br * blockCols + bc;
        for (unsigned cb = 0; cb < blockCol; ++cb, ++c) {
          if (sparsity[idxDense]) {
            vOutRowSparse[r].push_back(vOutRowMjSparse[idxElem]);
            vOutGradRowSparse[r].push_back(vOutGradRowMjSparse[idxElem]);
            ++idxElem;
          }
        }
      }
    }
  }

  std::vector<Tensor> nnOutTensors(rows);
  std::vector<Tensor> nnOutGradTensors(rows);
  for (unsigned r = 0; r < rows; ++r) {
    nnOutTensors[r] = graph.addConstant(
        outType, {1, vOutRowSparse[r].size()}, vOutRowSparse[r].data(),
        std::string("nnOut") + std::to_string(r));
    mapTensorLinearly(graph, nnOutTensors[r]);
    nnOutGradTensors[r] =
        graph.addConstant(outGradType, {1, vOutGradRowSparse[r].size()},
                          vOutGradRowSparse[r].data(),
                          std::string("nnOutGrad") + std::to_string(r));
    mapTensorLinearly(graph, nnOutGradTensors[r]);
  }

  std::vector<std::size_t> sparseShape = {nz, blockArea};
  Tensor bsOutTensor =
      graph.addConstant(outType, sparseShape, vOutBlockSparse.data(), "bsOutC");
  mapTensorLinearly(graph, bsOutTensor);
  Tensor bsOutGradTensor = graph.addConstant(
      outGradType, sparseShape, vOutGradBlockSparse.data(), "bsOutGrad");
  mapTensorLinearly(graph, bsOutGradTensor);

  Sequence softmaxProg;
  Tensor bsSoftmaxGrad = bsSoftmaxGradInternal(
      graph, bsOutTensor, bsOutGradTensor, blockRow, blockCol, blockRows,
      blockCols, sparsity.data(), softmaxProg, "bsSoftmaxGrad");
  BOOST_TEST(bsSoftmaxGrad.elementType() == outType);

  std::vector<Tensor> nnSoftmaxGrads(rows);
  for (unsigned int r = 0; r < rows; ++r) {
    nnSoftmaxGrads[r] = nonLinearityInputGradient(
        graph, NonLinearityType::SOFTMAX, nnOutTensors[r], nnOutGradTensors[r],
        softmaxProg, std::string("nnSoftmaxGrad") + std::to_string(r));
    BOOST_TEST(nnSoftmaxGrads[r].elementType() == outType);
  }

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  std::unique_ptr<char[]> bsSoftmaxGradRawHost =
      poplibs_test::util::allocateHostMemoryForTensor(
          bsSoftmaxGrad, "bsSoftmaxGrad", graph, uploadProg, downloadProg,
          streamMaps);

  std::vector<std::unique_ptr<char[]>> nnSoftmaxGradRawHosts(rows);
  for (unsigned int r = 0; r < rows; ++r) {
    nnSoftmaxGradRawHosts[r] = poplibs_test::util::allocateHostMemoryForTensor(
        nnSoftmaxGrads[r], std::string("nnSoftmaxGrad") + std::to_string(r),
        graph, uploadProg, downloadProg, streamMaps);
  }

  Sequence mainSequence;
  mainSequence.add(uploadProg);
  mainSequence.add(softmaxProg);
  mainSequence.add(
      poplar::program::PrintTensor("bsSoftmaxGrad", bsSoftmaxGrad));
  for (unsigned int r = 0; r < rows; ++r) {
    mainSequence.add(poplar::program::PrintTensor(
        std::string("nnSoftmaxGrads") + std::to_string(r), nnSoftmaxGrads[r]));
  }
  mainSequence.add(downloadProg);

  const OptionFlags engineOptions{{"target.workerStackSizeInBytes", "0x200"},
                                  {"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, mainSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  boost::multi_array<float, 2> bsSoftmaxGradHost(
      boost::extents[sparseShape[0]][sparseShape[1]]);
  poplibs_test::util::copy(target, outType, bsSoftmaxGradRawHost.get(),
                           bsSoftmaxGradHost);

  std::vector<boost::multi_array<float, 2>> nnSoftmaxGradHosts;
  for (unsigned int r = 0; r < rows; ++r) {
    boost::multi_array<float, 2> nnSoftmaxGradHost(
        boost::extents[1][vOutRowSparse[r].size()]);
    poplibs_test::util::copy(target, outType, nnSoftmaxGradRawHosts[r].get(),
                             nnSoftmaxGradHost);
    nnSoftmaxGradHosts.push_back(nnSoftmaxGradHost);
  }

  const float epsilon = (outType == FLOAT ? 0.001f : 0.1f);
  for (unsigned br = 0, idxBlock = 0; br < blockRows; ++br) {
    for (unsigned bcPacked = 0; bcPacked < nzBlocksByRow[br];
         ++bcPacked, ++idxBlock) {
      for (unsigned rb = 0; rb < blockRow; ++rb) {
        for (unsigned cb = 0; cb < blockCol; ++cb) {
          unsigned rowDense = br * blockRow + rb;
          unsigned colDense = bcPacked * blockCol + cb;
          unsigned idxInBlock = rb * blockCol + cb;
          // We compare the result only when softmaxrow is non-empty
          float valueBsSoftmaxGrad = bsSoftmaxGradHost[idxBlock][idxInBlock];
          float valueNnSoftmaxGrad;
          if (colDense < nnSoftmaxGradHosts[rowDense].shape()[1]) {
            valueNnSoftmaxGrad = nnSoftmaxGradHosts[rowDense][0][colDense];
          } else {
            valueNnSoftmaxGrad = 0.0f;
          }
          float err = fabs(valueBsSoftmaxGrad - valueNnSoftmaxGrad);
          if (err > epsilon) {
            BOOST_TEST(err <= epsilon);
          }
        }
      }
    }
  }
}

// Testing softmaxGrad()
/*
Block: 2x3

Dense block shape:
0 1 .
. 2 .
. 3 4
. . .

Sparse block shape:
0 1 2 3 4

Dense block shapes to compare:
0 1
2
3 4
.

*/
void softmaxGrad_test_3x3_4x3(const Type &outType, const Type &outGradType) {
  const unsigned blockRow = 2;
  const unsigned blockCol = 3;
  const unsigned blockRows = 4;
  const unsigned blockCols = 3;
  std::vector<unsigned char> sparsity(blockRows * blockCols, 0);
  sparsity[0 * blockCols + 0] = 1; // 0,0
  sparsity[0 * blockCols + 1] = 1; // 0,1
  sparsity[1 * blockCols + 1] = 1; // 1,1
  sparsity[2 * blockCols + 0] = 1; // 2,0
  sparsity[2 * blockCols + 1] = 1; // 2,1

  softmaxGradTest(blockRow, blockCol, blockRows, blockCols, sparsity, outType,
                  outGradType);
}

BOOST_AUTO_TEST_CASE(softmaxGrad_testF32_F32) {
  softmaxGrad_test_3x3_4x3(FLOAT, FLOAT);
}

BOOST_AUTO_TEST_CASE(softmaxGrad_testF16_F16) {
  softmaxGrad_test_3x3_4x3(HALF, HALF);
}