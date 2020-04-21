// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE BlockSparseTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <poplibs_test/Util.hpp>
#include <random>
#include <vector>

#include "popsparse/BSMatrix.hpp"
#include "popsparse/HyperGraph.hpp"
#include "popsparse/experimental/BlockSparseMatMul.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popsparse::experimental;

static const float MEMORY_CYCLE_RATIO = 1.0f;

// TODO: Move to some kind of utils
void getSparseMatrixBlocks(int rows, int cols, int rowsInBlock, int colsInBlock,
                           std::vector<unsigned char> &sparsity,
                           const boost::multi_array<float, 2> &denseMat,
                           boost::multi_array<float, 2> &blockData) {
  const int blockRows = rows / rowsInBlock;
  const int blockCols = cols / colsInBlock;

  int blockCount = 0;
  for (int br = 0; br < blockRows; br++) {
    for (int bc = 0; bc < blockCols; bc++) {
      if (sparsity[br * blockCols + bc] == 1) {
        blockCount++;
      }
    }
  }
  blockData.resize(boost::extents[blockCount][rowsInBlock * colsInBlock]);

  blockCount = 0;
  for (int br = 0; br < blockRows; br++) {
    for (int bc = 0; bc < blockCols; bc++) {
      if (sparsity[br * blockCols + bc] == 1) {
        int rowStart = br * rowsInBlock;
        int colStart = bc * colsInBlock;
        int rowEnd = rowStart + rowsInBlock;
        int colEnd = colStart + colsInBlock;
        int index = 0;
        for (int r = rowStart; r < rowEnd; r++) {
          for (int c = colStart; c < colEnd; c++) {
            blockData[blockCount][index++] = denseMat[r][c];
          }
        }
        blockCount++;
      }
    }
  }
}

boost::multi_array<float, 2> transpose(const boost::multi_array<float, 2> mat) {
  std::size_t rows = mat.shape()[0];
  std::size_t cols = mat.shape()[1];
  boost::multi_array<float, 2> transposedMat(boost::extents[cols][rows]);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      transposedMat[c][r] = mat[r][c];
    }
  }
  return transposedMat;
}

#define USE_RANDOM_VALUES 1

// TODO: Move to some kind of utils
void populateMatrixData1(boost::multi_array<float, 2> &host, int rows, int cols,
                         int rowsInBlock, int colsInBlock,
#if !USE_RANDOM_VALUES
                         int rowInBlockNz = 0, int colInBlockNz = 0,
                         int blockRowNz = -1, int blockColNz = -1
#else
                         int, int, int, int
#endif
) {

#if USE_RANDOM_VALUES
  std::mt19937 randomEngine;
  randomEngine.seed(102302);
#endif
  host.resize(boost::extents[rows][cols]);

  // We make only one element of a block as non-zero for simplicity
  // hostA is row-major
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
#if USE_RANDOM_VALUES
      host[r][c] =
          static_cast<float>(randomEngine()) / static_cast<float>(INT_MAX);
#else
      int rb = r % rowsInBlock;
      int br = r / rowsInBlock;
      int cb = c % colsInBlock;
      int bc = c / colsInBlock;
      if (rb == rowInBlockNz && cb == colInBlockNz &&
          (blockRowNz < 0 || br == blockRowNz) &&
          (blockColNz < 0 || bc == blockColNz)) {
        host[r][c] = 1.0f;
      } else {
        host[r][c] = 0.0f;
      }
#endif
    }
  }
}

void populateMatricesData1(boost::multi_array<float, 2> &hostA,
                           boost::multi_array<float, 2> &hostB, int rowsA,
                           int colsA, int colsB, int rowsInBlockA,
                           int colsInBlockA, int colsInBlockB,
                           int rowInBlockNzA = 0, int colInBlockNzA = 0,
                           int blockRowNzA = -1, int blockColNzA = -1) {

  const int rowsB = colsA;
  const int rowsInBlockB = colsInBlockA;

  hostA.resize(boost::extents[rowsA][colsA]);
  hostB.resize(boost::extents[rowsB][colsB]);

  const int rowInBlockNzB = colInBlockNzA;
  const int colInBlockNzB = rowInBlockNzA;
  const int blockRowNzB = blockColNzA;
  const int blockColNzB = blockRowNzA;

  populateMatrixData1(hostA, rowsA, colsA, rowsInBlockA, colsInBlockA,
                      rowInBlockNzA, colInBlockNzA, blockRowNzA, blockColNzA);

  populateMatrixData1(hostB, rowsB, colsB, rowsInBlockB, colsInBlockB,
                      rowInBlockNzB, colInBlockNzB, blockRowNzB, blockColNzB);
}

void populateMatricesData1(const std::array<int, 3> &dim,
                           const std::array<int, 3> &block_size,
                           boost::multi_array<float, 2> &lhsMatrix,
                           boost::multi_array<float, 2> &rhsMatrix,
                           const std::array<int, 2> &nzBlockElem,
                           const std::array<int, 2> &nzBlock) {
  populateMatricesData1(lhsMatrix, rhsMatrix, dim[0], dim[1], dim[2],
                        block_size[0], block_size[1], block_size[3],
                        nzBlockElem[0], nzBlockElem[1], nzBlock[0], nzBlock[1]);
}

/*
Computes matrix multiplication.
All matrices are row-major
*/
std::vector<std::vector<float>>
matMul(const boost::multi_array<float, 2> &hostA,
       const boost::multi_array<float, 2> &hostB) {
  std::size_t rowsA = hostA.shape()[0];
  std::size_t colsA = hostA.shape()[1];
  std::size_t colsB = hostB.shape()[1];
  std::vector<std::vector<float>> hostC;
  for (std::size_t r = 0; r < rowsA; ++r) {
    hostC.push_back({});
    for (std::size_t c = 0; c < colsB; ++c) {
      float sum = 0.0f;
      for (std::size_t mulc = 0; mulc < colsA; ++mulc) {
        sum += hostA[r][mulc] * hostB[mulc][c];
      }
      hostC[r].push_back(sum);
    }
  }
  return hostC;
}

void checkDenseResult(
    const poplar::Type &dataType, int blockRowsC, int blockColsC,
    int rowsInBlockC, int colsInBlockC,
    const std::vector<std::vector<float>> &hostC,
    const std::vector<boost::multi_array<float, 2>> &blocksHostCParts) {
  const float epsilon = (dataType == FLOAT ? 0.001f : 0.1f);
  for (int br = 0, r = 0; br < blockRowsC; ++br) {
    for (int rb = 0; rb < rowsInBlockC; ++rb, ++r) {
      for (int bc = 0, c = 0; bc < blockColsC; ++bc) {
        int outBlockIdx = br * blockColsC + bc;
        for (int cb = 0; cb < colsInBlockC; ++cb, ++c) {
          float val = blocksHostCParts[outBlockIdx][rb][cb];
          float err = fabs(val - hostC[r][c]);
          if (err > epsilon)
            BOOST_TEST(err <= epsilon);
        }
      }
    }
  }
}

void checkSparseResult(
    const poplar::Type &dataType, int blockRowsC, int blockColsC,
    int rowsInBlockC, int colsInBlockC,
    const std::vector<std::vector<float>> &hostC,
    const std::vector<boost::multi_array<float, 2>> &blocksHostCParts,
    const std::vector<unsigned char> &sparsityC) {
  const float epsilon = (dataType == FLOAT ? 0.001f : 0.05f);
  for (int br = 0, r = 0; br < blockRowsC; ++br) {
    for (int rb = 0; rb < rowsInBlockC; ++rb, ++r) {
      for (int bc = 0, c = 0; bc < blockColsC; ++bc) {
        int outBlockIdx = br * blockColsC + bc;
        for (int cb = 0; cb < colsInBlockC; ++cb, ++c) {
          float val = blocksHostCParts[outBlockIdx][rb][cb];
          float valTruth = hostC[r][c];
          if (sparsityC[br * blockColsC + bc] == 0) {
            valTruth = 0.0f;
          }
          float err = fabs(val - valTruth);
          if (err > epsilon)
            BOOST_TEST(err <= epsilon);
        }
      }
    }
  }
}

/*
Testing BlockSparseMatrix constructor and getBlockIdMatrix() method
*/
BOOST_AUTO_TEST_CASE(BlockSparseMatrix_test) {
  const int rows = 6;
  const int cols = 6;
  const int rows_block = 2;
  const int cols_block = 2;
  const int block_rows = rows / rows_block;
  const int block_cols = cols / cols_block;
  std::vector<unsigned char> sparsity(block_rows * block_cols, 0);
  sparsity[0 + 1] = 1;
  sparsity[3 + 0] = 1;
  sparsity[3 + 2] = 1;
  sparsity[6 + 1] = 1;

  BlockSparseMatrix bs1(rows, cols, rows_block, cols_block, false,
                        sparsity.data());

  std::vector<std::vector<int>> blockIdMatrix = bs1.getBlockIdMatrix();

  std::vector<std::vector<int>> blockIdMatrix_expected(3,
                                                       std::vector<int>(3, -1));
  blockIdMatrix_expected[0][1] = 0;
  blockIdMatrix_expected[1][0] = 1;
  blockIdMatrix_expected[1][2] = 2;
  blockIdMatrix_expected[2][1] = 3;

  BOOST_TEST(blockIdMatrix == blockIdMatrix_expected);
}

/*
Testing Hypergraph for MatMul nodes and edges - no reduction case

nodeA_   nodeB_   nodeC_
0 1    x 4 5    =  8  9
2 3      6 7      10 11

nodeV_
id  idxA_ idxB_
12  [0,1] [0,2]
13  [0,1] [1,3]
14  [2,3] [0,2]
15  [2,3] [1,3]

*/
BOOST_AUTO_TEST_CASE(HyperGraph_testMatMulNoReduction) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const int rows = 2;
  const int cols = 2;
  const int rows_block = 1;
  const int cols_block = 1;
  const int block_rows = rows / rows_block;
  const int block_cols = cols / cols_block;

  std::vector<unsigned char> sparsityB(block_rows * block_cols, 1);

  BlockDenseMatrix A(rows, cols, rows_block, cols_block, false);

  BlockSparseMatrix B(rows, cols, rows_block, cols_block, false,
                      sparsityB.data());

  // No reduction - 1 tile
  HyperGraph hg(A, B, FLOAT, FLOAT, FLOAT, 1, 1);

  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  auto &nodeA = hg.getNodeA();
  auto &nodeB = hg.getNodeB();
  auto &nodeC = hg.getNodeC();
  auto &nodeV = hg.getNodeV();

  auto &edgeA = hg.getEdgeA();
  auto &edgeB = hg.getEdgeB();
  auto &edgeC = hg.getEdgeC();

  BOOST_TEST(nodeA.size() == 4U);
  BOOST_TEST(nodeB.size() == 4U);
  BOOST_TEST(nodeC.size() == 4U);
  BOOST_TEST(nodeV.size() == 4U);

  BOOST_TEST(edgeA.size() == 4U);
  BOOST_TEST(edgeB.size() == 4U);
  BOOST_TEST(edgeC.size() == 4U);

  BOOST_TEST(nodeV[0].idxA.size() == 2U);
  BOOST_TEST(nodeV[0].idxB.size() == 2U);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[1].idxA.size() == 2U);
  BOOST_TEST(nodeV[1].idxB.size() == 2U);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(nodeV[2].idxA.size() == 2U);
  BOOST_TEST(nodeV[2].idxB.size() == 2U);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[3].idxA.size() == 2U);
  BOOST_TEST(nodeV[3].idxB.size() == 2U);
  BOOST_TEST(nodeV[3].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[3].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(edgeA[0].in.size() == 1U);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2U);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[1].in.size() == 1U);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2U);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[2].in.size() == 1U);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 2U);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeA[3].in.size() == 1U);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 2U);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeB[0].in.size() == 1U);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 2U);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeB[1].in.size() == 1U);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2U);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeB[2].in.size() == 1U);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 2U);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeB[3].in.size() == 1U);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2U);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeC[0].out.size() == 1U);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 1U);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeC[1].out.size() == 1U);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 1U);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[1].id}));

  BOOST_TEST(edgeC[2].out.size() == 1U);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 1U);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeC[3].out.size() == 1U);
  BOOST_TEST(edgeC[3].out == std::vector<unsigned>({nodeC[3].id}));
  BOOST_TEST(edgeC[3].in.size() == 1U);
  BOOST_TEST(edgeC[3].in == std::vector<unsigned>({nodeV[3].id}));
}

/*
Testing Hypergraph for MatMul nodes and edges - reduction case

nodeA_   nodeB_   nodeC_
0 1    x 4 5    =  8  9
2 3      6 7      10 11

nodeV_
id  idxA_ idxB_
12  [0] [0]
13  [1] [2]
14  [0] [1]
15  [1] [3]
16  [2] [0]
17  [3] [2]
18  [2] [1]
19  [3] [3]

*/
BOOST_AUTO_TEST_CASE(HyperGraph_testMatMulWReduction) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const int rows = 2;
  const int cols = 2;
  const int rows_block = 1;
  const int cols_block = 1;
  const int block_rows = rows / rows_block;
  const int block_cols = cols / cols_block;

  std::vector<unsigned char> sparsityB(block_rows * block_cols, 1);

  BlockDenseMatrix A(rows, cols, rows_block, cols_block, false);

  BlockSparseMatrix B(rows, cols, rows_block, cols_block, false,
                      sparsityB.data());

  // Reduction - 1216 tiles
  HyperGraph hg(A, B, FLOAT, FLOAT, FLOAT, 1216);

  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  auto &nodeA = hg.getNodeA();
  auto &nodeB = hg.getNodeB();
  auto &nodeC = hg.getNodeC();
  auto &nodeV = hg.getNodeV();

  auto &edgeA = hg.getEdgeA();
  auto &edgeB = hg.getEdgeB();
  auto &edgeC = hg.getEdgeC();

  BOOST_TEST(nodeA.size() == 4U);
  BOOST_TEST(nodeB.size() == 4U);
  BOOST_TEST(nodeC.size() == 4U);
  BOOST_TEST(nodeV.size() == 8U);

  BOOST_TEST(edgeA.size() == 4U);
  BOOST_TEST(edgeB.size() == 4U);
  BOOST_TEST(edgeC.size() == 4U);

  BOOST_TEST(nodeV[0].idxA.size() == 1U);
  BOOST_TEST(nodeV[0].idxB.size() == 1U);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0}));

  BOOST_TEST(nodeV[1].idxA.size() == 1U);
  BOOST_TEST(nodeV[1].idxB.size() == 1U);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({2}));

  BOOST_TEST(nodeV[2].idxA.size() == 1U);
  BOOST_TEST(nodeV[2].idxB.size() == 1U);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({0}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({1}));

  BOOST_TEST(nodeV[3].idxA.size() == 1U);
  BOOST_TEST(nodeV[3].idxB.size() == 1U);
  BOOST_TEST(nodeV[3].idxA == std::vector<unsigned>({1}));
  BOOST_TEST(nodeV[3].idxB == std::vector<unsigned>({3}));

  BOOST_TEST(nodeV[4].idxA.size() == 1U);
  BOOST_TEST(nodeV[4].idxB.size() == 1U);
  BOOST_TEST(nodeV[4].idxA == std::vector<unsigned>({2}));
  BOOST_TEST(nodeV[4].idxB == std::vector<unsigned>({0}));

  BOOST_TEST(nodeV[5].idxA.size() == 1U);
  BOOST_TEST(nodeV[5].idxB.size() == 1U);
  BOOST_TEST(nodeV[5].idxA == std::vector<unsigned>({3}));
  BOOST_TEST(nodeV[5].idxB == std::vector<unsigned>({2}));

  BOOST_TEST(nodeV[6].idxA.size() == 1U);
  BOOST_TEST(nodeV[6].idxB.size() == 1U);
  BOOST_TEST(nodeV[6].idxA == std::vector<unsigned>({2}));
  BOOST_TEST(nodeV[6].idxB == std::vector<unsigned>({1}));

  BOOST_TEST(nodeV[7].idxA.size() == 1U);
  BOOST_TEST(nodeV[7].idxB.size() == 1U);
  BOOST_TEST(nodeV[7].idxA == std::vector<unsigned>({3}));
  BOOST_TEST(nodeV[7].idxB == std::vector<unsigned>({3}));

  BOOST_TEST(edgeA[0].in.size() == 1U);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2U);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeA[1].in.size() == 1U);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2U);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeA[2].in.size() == 1U);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 2U);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[4].id, nodeV[6].id}));

  BOOST_TEST(edgeA[3].in.size() == 1U);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 2U);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[5].id, nodeV[7].id}));

  BOOST_TEST(edgeB[0].in.size() == 1U);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 2U);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[4].id}));

  BOOST_TEST(edgeB[1].in.size() == 1U);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2U);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[2].id, nodeV[6].id}));

  BOOST_TEST(edgeB[2].in.size() == 1U);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 2U);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[1].id, nodeV[5].id}));

  BOOST_TEST(edgeB[3].in.size() == 1U);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2U);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[3].id, nodeV[7].id}));

  BOOST_TEST(edgeC[0].out.size() == 1U);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 2U);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeC[1].out.size() == 1U);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 2U);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeC[2].out.size() == 1U);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 2U);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[4].id, nodeV[5].id}));

  BOOST_TEST(edgeC[3].out.size() == 1U);
  BOOST_TEST(edgeC[3].out == std::vector<unsigned>({nodeC[3].id}));
  BOOST_TEST(edgeC[3].in.size() == 2U);
  BOOST_TEST(edgeC[3].in == std::vector<unsigned>({nodeV[6].id, nodeV[7].id}));
}

/*
Testing Hypergraph for MatMulOuter nodes and edges - no reduction case

nodeA_   nodeB_   nodeC_
0 1    x 4 5    =  8  9
2 3      6 7      x  10

nodeV_
id  idxA_ idxB_
12  [0,1] [0,2]
13  [0,1] [1,3]
14  [2,3] [1,3]

*/
BOOST_AUTO_TEST_CASE(nColC) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const int rows = 2;
  const int cols = 2;
  const int rows_block = 1;
  const int cols_block = 1;
  const int block_rows = rows / rows_block;
  const int block_cols = cols / cols_block;

  BlockDenseMatrix A(rows, cols, rows_block, cols_block, false);

  BlockDenseMatrix B(rows, cols, rows_block, cols_block, false);

  std::vector<unsigned char> sparsityC(block_rows * block_cols, 1);
  sparsityC[2 + 0] = 0;

  // No reduction - 1 tile
  HyperGraph hg(A, B, FLOAT, FLOAT, FLOAT, 1, 1);

  hg.createGraphMatMulSparsifyResult(sparsityC.data(), MEMORY_CYCLE_RATIO,
                                     graph, "C");

  auto &nodeA = hg.getNodeA();
  auto &nodeB = hg.getNodeB();
  auto &nodeC = hg.getNodeC();
  auto &nodeV = hg.getNodeV();

  auto &edgeA = hg.getEdgeA();
  auto &edgeB = hg.getEdgeB();
  auto &edgeC = hg.getEdgeC();

  BOOST_TEST(nodeA.size() == 4U);
  BOOST_TEST(nodeB.size() == 4U);
  BOOST_TEST(nodeC.size() == 3U);
  BOOST_TEST(nodeV.size() == 3U);

  BOOST_TEST(edgeA.size() == 4U);
  BOOST_TEST(edgeB.size() == 4U);
  BOOST_TEST(edgeC.size() == 3U);

  BOOST_TEST(nodeV[0].idxA.size() == 2U);
  BOOST_TEST(nodeV[0].idxB.size() == 2U);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[1].idxA.size() == 2U);
  BOOST_TEST(nodeV[1].idxB.size() == 2U);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(nodeV[2].idxA.size() == 2U);
  BOOST_TEST(nodeV[2].idxB.size() == 2U);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(edgeA[0].in.size() == 1U);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2U);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[1].in.size() == 1U);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2U);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[2].in.size() == 1U);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 1U);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeA[3].in.size() == 1U);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 1U);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeB[0].in.size() == 1U);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 1U);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeB[1].in.size() == 1U);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2U);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[2].id}));

  BOOST_TEST(edgeB[2].in.size() == 1U);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 1U);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeB[3].in.size() == 1U);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2U);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[1].id, nodeV[2].id}));

  BOOST_TEST(edgeC[0].out.size() == 1U);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 1U);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeC[1].out.size() == 1U);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 1U);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[1].id}));

  BOOST_TEST(edgeC[2].out.size() == 1U);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 1U);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[2].id}));
}

void TestMatMul(const poplar::Type &dataType, int blockSize, int batchBlockSize,
                int blockRowsA, int blockColsA, int blockColsB,
                int rowInBlockNzA = 0, int colInBlockNzA = 0,
                int blockRowNzA = -1, int blockColNzA = -1);

/*
Testing asymmetric layout of blocks and asymmetric layout within a block
*/
BOOST_AUTO_TEST_CASE(MatMul_testF32_asymmetric) {
  TestMatMul(FLOAT, 8, 6, 2, 2, 2, 1, 2, 0, 1);
}

/*
Testing batch block of 8
*/
BOOST_AUTO_TEST_CASE(MatMul_testF32_batchBlock8) {
  TestMatMul(FLOAT, 8, 8, 2, 2, 2, 1, 2, 0, 1);
}

/*
Testing all combinations or number of block rows and columns from 1 to 2
*/
BOOST_AUTO_TEST_CASE(MatMul_testF32) {
  for (int blockRowsA = 1; blockRowsA <= 2; ++blockRowsA) {
    for (int blockColsA = 1; blockColsA <= 2; ++blockColsA) {
      for (int blockColsB = 1; blockColsB <= 2; ++blockColsB) {
        TestMatMul(FLOAT, 8, 6, blockRowsA, blockColsA, blockColsB, 1, 2);
      }
    }
  }
}

/*
Testing fp16
*/
BOOST_AUTO_TEST_CASE(MatMul_testF16) { TestMatMul(HALF, 16, 6, 1, 2, 1); }

/*
Testing MatMul vertex - no reduction case
*/
void TestMatMul(const poplar::Type &dataType, int blockSize, int batchBlockSize,
                int blockRowsA, int blockColsA, int blockColsB,
                int rowInBlockNzA, int colInBlockNzA, int blockRowNzA,
                int blockColNzA) {

  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  HyperGraph::addCodelets(graph);

  const int rowsInBlockA = batchBlockSize;
  const int colsInBlockA = blockSize;
  const int rowsInBlockB = colsInBlockA;
  const int colsInBlockB = blockSize;
  const int rowsInBlockC = rowsInBlockA;
  const int colsInBlockC = colsInBlockB;

  const int blockRowsB = blockColsA;
  const int blockRowsC = blockRowsA;
  const int blockColsC = blockColsB;

  const int rowsA = rowsInBlockA * blockRowsA;
  const int colsA = colsInBlockA * blockColsA;
  const int rowsB = rowsInBlockB * blockRowsB;
  const int colsB = colsInBlockB * blockColsB;

  unsigned outBlocks = blockRowsC * blockColsC;

  boost::multi_array<float, 2> hostA;
  boost::multi_array<float, 2> hostB;
  populateMatricesData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                        colsInBlockA, colsInBlockB, rowInBlockNzA,
                        colInBlockNzA, blockRowNzA, blockColNzA);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> hostBTransposed = transpose(hostB);

  // Blocks layout in blocksHostB are row-major
  // Individual blocks are column-major
  boost::multi_array<float, 2> blocksHostB;
  // Transpose B
  getSparseMatrixBlocks(colsB, rowsB, colsInBlockB, rowsInBlockB, sparsityB,
                        hostBTransposed, blocksHostB);

  BlockDenseMatrix A(rowsA, colsA, rowsInBlockA, colsInBlockA, false);

  BlockSparseMatrix B(colsB, rowsB, colsInBlockB, rowsInBlockB, true,
                      sparsityB.data());

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  // LHS dense matrix
  poplar::Tensor tensorA = A.createTensor(graph, dataType, "A");
  A.setBlockTensor(tensorA);

  std::unique_ptr<char[]> rawHostA =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA, "A", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA, dataType, rawHostA.get());

  // RHS sparse matrix
  poplar::Tensor tensorB = B.createTensor(graph, dataType, "B");
  B.setBlockTensor(tensorB);

  std::unique_ptr<char[]> blocksRawHostB =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB, "B", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, blocksHostB, dataType, blocksRawHostB.get());

  // No reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, 1, 1);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  // Put everything on tile 0
  std::vector<int> tileAssignment(hg.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  std::size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == outBlocks);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;
  for (auto iter = tensorCParts.begin(); iter != tensorCParts.end(); ++iter) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (std::size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  // hostC is row-major
  std::vector<std::vector<float>> hostC = matMul(hostA, hostB);

  checkDenseResult(dataType, blockRowsC, blockColsC, rowsInBlockC, colsInBlockC,
                   hostC, blocksHostCParts);
}

void TestMatMulReduce(const poplar::Type &dataType, int blockSize,
                      int batchBlockSize);

BOOST_AUTO_TEST_CASE(MatMulReduce_testF32) { TestMatMulReduce(FLOAT, 8, 6); }

BOOST_AUTO_TEST_CASE(MatMulReduce_testF32batchBlock8) {
  TestMatMulReduce(FLOAT, 8, 8);
}

/*
Testing MatMul and Reduce vertices
*/
void TestMatMulReduce(const poplar::Type &dataType, int blockSize,
                      int batchSize) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  HyperGraph::addCodelets(graph);

  const int blockRowsA = 1;
  const int blockColsA = 2;
  const int blockColsB = 1;

  const int rowsInBlockA = batchSize;
  const int colsInBlockA = blockSize;
  const int rowsInBlockB = colsInBlockA;
  const int colsInBlockB = blockSize;
  const int rowsInBlockC = rowsInBlockA;
  const int colsInBlockC = colsInBlockB;

  const int blockRowsB = blockColsA;
  const int blockRowsC = blockRowsA;
  const int blockColsC = blockColsB;

  const int rowsA = rowsInBlockA * blockRowsA;
  const int colsA = colsInBlockA * blockColsA;
  const int rowsB = rowsInBlockB * blockRowsB;
  const int colsB = colsInBlockB * blockColsB;

  boost::multi_array<float, 2> hostA;
  boost::multi_array<float, 2> hostB;
  populateMatricesData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                        colsInBlockA, colsInBlockB, 1, 2);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> hostBTransposed = transpose(hostB);

  boost::multi_array<float, 2> blocksHostB;
  // Transpose B
  getSparseMatrixBlocks(colsB, rowsB, colsInBlockB, rowsInBlockB, sparsityB,
                        hostBTransposed, blocksHostB);

  BlockDenseMatrix A(rowsA, colsA, rowsInBlockA, colsInBlockA, false);

  BlockSparseMatrix B(colsB, rowsB, colsInBlockB, rowsInBlockB, true,
                      sparsityB.data());

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  // LHS dense matrix
  poplar::Tensor tensorA = A.createTensor(graph, dataType, "A");
  A.setBlockTensor(tensorA);

  std::unique_ptr<char[]> rawHostA =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA, "A", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA, dataType, rawHostA.get());

  // RHS sparse matrix
  poplar::Tensor tensorB = B.createTensor(graph, dataType, "B");
  B.setBlockTensor(tensorB);

  std::unique_ptr<char[]> blocksRawHostB =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB, "B", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, blocksHostB, dataType, blocksRawHostB.get());

  // Reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, ipuModel.tilesPerIPU);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  // Put everything on tile 0
  std::vector<int> tileAssignment(hg.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  std::size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == hg.getNodeV().size());

  poplar::program::Sequence reduceProg;
  hg.createComputeSetReduce(tensorCParts, nodeCTileId, graph, "test",
                            reduceProg);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;

  for (std::size_t i = 0; i < hg.matC->getBlockTensor().size(); ++i) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        hg.matC->getBlockTensor()[i], std::string("C_") + std::to_string(i),
        graph, uploadProg, downloadProg, streamMaps));
  }

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(reduceProg);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (std::size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  std::vector<std::vector<float>> hostC = matMul(hostA, hostB);

  checkDenseResult(dataType, blockRowsC, blockColsC, rowsInBlockC, colsInBlockC,
                   hostC, blocksHostCParts);
}

void TestMatMulOuter(const poplar::Type &dataType, bool needTranspose);

BOOST_AUTO_TEST_CASE(MatMulOuter_testF32) { TestMatMulOuter(FLOAT, false); }

BOOST_AUTO_TEST_CASE(MatMulOuterTranspose_testF32) {
  TestMatMulOuter(FLOAT, true);
}

/*
Testing MatMulOuter vertex - no reduction case
*/
void TestMatMulOuter(const poplar::Type &dataType, bool needTranspose) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  HyperGraph::addCodelets(graph);

  const int blockSize = 8;
  const int batchSize = 6;

  const int blockRowsA = 2;
  const int blockColsA = 2;
  const int blockColsB = 2;

  const int rowsInBlockA = batchSize;
  const int colsInBlockA = blockSize;
  const int rowsInBlockB = colsInBlockA;
  const int colsInBlockB = blockSize;
  const int rowsInBlockC = rowsInBlockA;
  const int colsInBlockC = colsInBlockB;

  const int blockRowsB = blockColsA;
  const int blockRowsC = blockRowsA;
  const int blockColsC = blockColsB;

  const int rowsA = rowsInBlockA * blockRowsA;
  const int colsA = colsInBlockA * blockColsA;
  const int rowsB = rowsInBlockB * blockRowsB;
  const int colsB = colsInBlockB * blockColsB;

  unsigned outBlocks = blockRowsC * blockColsC;

  boost::multi_array<float, 2> hostA;
  boost::multi_array<float, 2> hostB;
  populateMatricesData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                        colsInBlockA, colsInBlockB, 1, 2, 0, 1);

  std::vector<unsigned char> sparsityC(blockRowsC * blockColsC, 1);

  // Mask [1][0] element out
  const int blockRowZero = 1;
  const int blockColZero = 0;
  sparsityC[blockRowZero * blockColsB + blockColZero] = 0;
  --outBlocks;

  BlockDenseMatrix A(rowsA, colsA, rowsInBlockA, colsInBlockA, false);

  BlockDenseMatrix B(colsB, rowsB, colsInBlockB, rowsInBlockB, needTranspose);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  // LHS dense matrix
  poplar::Tensor tensorA = A.createTensor(graph, dataType, "A");
  A.setBlockTensor(tensorA);

  std::unique_ptr<char[]> rawHostA =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA, "A", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA, dataType, rawHostA.get());

  // RHS dense matrix
  poplar::Tensor tensorB = B.createTensor(graph, dataType, "B");
  B.setBlockTensor(tensorB);

  std::unique_ptr<char[]> rawHostB =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB, "B", graph, uploadProg, downloadProg, streamMaps);
  if (needTranspose) {
    boost::multi_array<float, 2> hostBTransposed = transpose(hostB);
    poplibs_test::util::copy(target, hostBTransposed, dataType, rawHostB.get());
  } else {
    poplibs_test::util::copy(target, hostB, dataType, rawHostB.get());
  }

  // No reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, 1, 1);
  hg.createGraphMatMulSparsifyResult(sparsityC.data(), MEMORY_CYCLE_RATIO,
                                     graph, "C");

  // Put everything on tile 0
  std::vector<int> tileAssignment(hg.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  std::size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == outBlocks);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;
  for (auto iter = tensorCParts.begin(); iter != tensorCParts.end(); ++iter) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (int br = 0, outBlockIdx = 0; br < blockRowsC; ++br) {
    for (int bc = 0; bc < blockColsC; ++bc) {
      boost::multi_array<float, 2> blocksHostCPart(
          boost::extents[rowsInBlockC][colsInBlockC]);
      std::fill_n(blocksHostCPart.data(), blocksHostCPart.num_elements(), 0.0f);
      if (sparsityC[br * blockColsC + bc] > 0) {
        poplibs_test::util::copy(target, dataType,
                                 rawHostCParts[outBlockIdx].get(),
                                 blocksHostCPart);
        ++outBlockIdx;
      }
      blocksHostCParts.push_back(std::move(blocksHostCPart));
    }
  }

  std::vector<std::vector<float>> hostC = matMul(hostA, hostB);

  checkSparseResult(dataType, blockRowsC, blockColsC, rowsInBlockC,
                    colsInBlockC, hostC, blocksHostCParts, sparsityC);
}

void TestSparseTensorReuse4Transpose(const poplar::Type &dataType,
                                     int blockSize, int blockRowsA,
                                     int blockColsA, int blockColsB,
                                     int rowInBlockNzA, int colInBlockNzA,
                                     int blockRowNzA, int blockColNzA);

BOOST_AUTO_TEST_CASE(SparseTensorReuse4Transpose_testF32) {
  TestSparseTensorReuse4Transpose(FLOAT, 8, 2, 3, 2, 1, 2, 0, 1);
}

/*
Testing that created by API sparse tensor can be used for both transposed and
non-transposed scenarios

  1) X * W = Y

  2) dY * Wt = dX

  Using notations:

  X == A, W == B, Y == C
  dX == A1, Wt == Bt, dY == C1

*/
void TestSparseTensorReuse4Transpose(const poplar::Type &dataType,
                                     int blockSize, int blockRowsA,
                                     int blockColsA, int blockColsB,
                                     int rowInBlockNzA, int colInBlockNzA,
                                     int blockRowNzA, int blockColNzA) {

  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  HyperGraph::addCodelets(graph);

  const int batchSize = 6;

  const int rowsInBlockA = batchSize;
  const int colsInBlockA = blockSize;
  const int rowsInBlockB = colsInBlockA;
  const int colsInBlockB = blockSize;
  const int rowsInBlockC = rowsInBlockA;
  const int colsInBlockC = colsInBlockB;

  ///////////////////
  // A * B = C
  const int blockRowsB = blockColsA;
  const int blockRowsC = blockRowsA;
  const int blockColsC = blockColsB;

  const int rowsA = rowsInBlockA * blockRowsA;
  const int colsA = colsInBlockA * blockColsA;
  const int rowsB = rowsInBlockB * blockRowsB;
  const int colsB = colsInBlockB * blockColsB;

  unsigned outBlocks = blockRowsC * blockColsC;

  boost::multi_array<float, 2> hostA;
  boost::multi_array<float, 2> hostB;
  populateMatricesData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                        colsInBlockA, colsInBlockB, rowInBlockNzA,
                        colInBlockNzA, blockRowNzA, blockColNzA);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> blocksHostB;
  getSparseMatrixBlocks(rowsB, colsB, rowsInBlockB, colsInBlockB, sparsityB,
                        hostB, blocksHostB);

  BlockDenseMatrix A(rowsA, colsA, rowsInBlockA, colsInBlockA, false);

  BlockSparseMatrix B(rowsB, colsB, colsInBlockB, rowsInBlockB, false,
                      sparsityB.data());

  ///////////////////
  // A1 * Bt = C1
  const int blockRowsA1 = blockRowsA;
  const int blockColsA1 = blockRowsC;

  const int blockRowsC1 = blockRowsA1;
  const int blockColsC1 = blockRowsB;

  const int rowsInBlockA1 = rowsInBlockA;
  const int colsInBlockA1 = colsInBlockC;

  const int rowsA1 = rowsInBlockA1 * blockRowsA1;
  const int colsA1 = colsInBlockA1 * blockColsA1;

  unsigned outBlocks1 = blockRowsC1 * blockColsC1;

  boost::multi_array<float, 2> hostA1;
  populateMatrixData1(hostA1, rowsA1, colsA1, rowsInBlockA1, colsInBlockA1,
                      // Swap rows and columns to match transposed B
                      colInBlockNzA, rowInBlockNzA, blockColNzA, blockRowNzA);

  boost::multi_array<float, 2> hostBt = transpose(hostB);

  std::vector<unsigned char> sparsityBt(blockColsB * blockRowsB, 1);

  BlockDenseMatrix A1(rowsA1, colsA1, rowsInBlockA1, colsInBlockA1, false);

  // Bt[colsB x rowsB] - here rows and cols are swapped again, because of "true"
  // flag
  BlockSparseMatrix Bt(rowsB, colsB, rowsInBlockB, colsInBlockB, true,
                       sparsityBt.data());

  ///////////////////
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  /////////////////// A * B = C
  poplar::Tensor tensorA = A.createTensor(graph, dataType, "A");
  A.setBlockTensor(tensorA);

  std::unique_ptr<char[]> rawHostA =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA, "A", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA, dataType, rawHostA.get());

  poplar::Tensor tensorB = B.createTensor(graph, dataType, "B");
  B.setBlockTensor(tensorB);

  std::unique_ptr<char[]> blocksRawHostB =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB, "B", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, blocksHostB, dataType, blocksRawHostB.get());

  // No reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, 1, 1);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  std::vector<int> tileAssignment(hg.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  std::size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == outBlocks);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;
  for (auto iter = tensorCParts.begin(); iter != tensorCParts.end(); ++iter) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }

  /////////////////// A1 * Bt = C1
  poplar::Tensor tensorA1 = A1.createTensor(graph, dataType, "A1");
  A1.setBlockTensor(tensorA1);

  std::unique_ptr<char[]> rawHostA1 =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA1, "A1", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA1, dataType, rawHostA1.get());

#if 0 // Reference path for tensor Bt

  boost::multi_array<float, 2> blocksHostBt;
  // Here rows and cols are swapped again, because hostB is not tranposed
  getSparseMatrixBlocks(rowsB, colsB, rowsInBlockB, colsInBlockB, sparsityBt,
                        hostB, blocksHostBt);

  poplar::Tensor tensorBt = Bt.createTensor(graph, dataType, "Bt");
  Bt.setBlockTensor(tensorBt);

  std::unique_ptr<char[]> blocksRawHostBt = poplibs_test::util::allocateHostMemoryForTensor(
        tensorBt, "Bt", graph, uploadProg,
        downloadProg, streamMaps);
  poplibs_test::util::copy(target, blocksHostBt, dataType,
                             blocksRawHostBt.get());

#else

  // Reusing existing tensor in transposed matrix
  Bt.setBlockTensor(tensorB);

#endif

  // No reduction
  HyperGraph hg1(A1, Bt, dataType, dataType, dataType, 1, 1);
  hg1.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C1");

  std::vector<int> tileAssignment1(hg1.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts1;
  std::vector<unsigned int> nodeCTileId1;
  poplar::program::Sequence matMulProg1;
  hg1.createComputeSetMatMul(tileAssignment1, tensorCParts1, nodeCTileId1,
                             graph, "matMul1", matMulProg1);
  std::size_t tensorCPartsLen1 = tensorCParts1.size();
  BOOST_TEST(tensorCPartsLen1 == outBlocks1);

  std::vector<std::unique_ptr<char[]>> rawHostC1Parts;
  for (auto iter = tensorCParts1.begin(); iter != tensorCParts1.end(); ++iter) {
    rawHostC1Parts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C1_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }
  ///////////////////

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(matMulProg1);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  /////////////////// C
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (std::size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  std::vector<std::vector<float>> hostC = matMul(hostA, hostB);

  checkDenseResult(dataType, blockRowsC, blockColsC, rowsInBlockC, colsInBlockC,
                   hostC, blocksHostCParts);

  /////////////////// C1
  std::vector<boost::multi_array<float, 2>> blocksHostC1Parts;
  for (std::size_t i = 0; i < rawHostC1Parts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostC1Part(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostC1Parts[i].get(),
                             blocksHostC1Part);
    blocksHostC1Parts.push_back(std::move(blocksHostC1Part));
  }

  std::vector<std::vector<float>> hostC1 = matMul(hostA1, hostBt);

  checkDenseResult(dataType, blockRowsC1, blockColsC1, rowsInBlockC,
                   colsInBlockC, hostC1, blocksHostC1Parts);
}

void TestDenseTensorReuse4Transpose(const poplar::Type &dataType, int blockSize,
                                    int blockRowsA, int blockColsA,
                                    int blockColsB, int rowInBlockNzA,
                                    int colInBlockNzA, int blockRowNzA,
                                    int blockColNzA);

BOOST_AUTO_TEST_CASE(DenseTensorReuse4Transpose_testF32) {
  TestDenseTensorReuse4Transpose(FLOAT, 8, 2, 3, 2, 1, 2, 0, 1);
}

/*
Testing that created by API dense tensor can be used for both transposed and
non-transposed scenarios

  1) X * W = Y

  2) Xt * dY = dW

  Using notations:

  X == A, W == B, Y == C
  Xt == At, dY == B1, Y == C1

*/
void TestDenseTensorReuse4Transpose(const poplar::Type &dataType, int blockSize,
                                    int blockRowsA, int blockColsA,
                                    int blockColsB, int rowInBlockNzA,
                                    int colInBlockNzA, int blockRowNzA,
                                    int blockColNzA) {

  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);
  HyperGraph::addCodelets(graph);

  const int batchSize = 8;

  const int rowsInBlockA = batchSize;
  const int colsInBlockA = blockSize;
  const int rowsInBlockB = colsInBlockA;
  const int colsInBlockB = blockSize;
  const int rowsInBlockC = rowsInBlockA;
  const int colsInBlockC = colsInBlockB;

  /////////////////// A * B = C
  //
  const int blockRowsB = blockColsA;
  const int blockRowsC = blockRowsA;
  const int blockColsC = blockColsB;

  const int rowsA = rowsInBlockA * blockRowsA;
  const int colsA = colsInBlockA * blockColsA;
  const int rowsB = rowsInBlockB * blockRowsB;
  const int colsB = colsInBlockB * blockColsB;

  unsigned outBlocks = blockRowsC * blockColsC;

  boost::multi_array<float, 2> hostA;
  boost::multi_array<float, 2> hostB;
  populateMatricesData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                        colsInBlockA, colsInBlockB, rowInBlockNzA,
                        colInBlockNzA, blockRowNzA, blockColNzA);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> blocksHostB;
  getSparseMatrixBlocks(rowsB, colsB, rowsInBlockB, colsInBlockB, sparsityB,
                        hostB, blocksHostB);

  BlockDenseMatrix A(rowsA, colsA, rowsInBlockA, colsInBlockA, false);

  BlockSparseMatrix B(rowsB, colsB, colsInBlockB, rowsInBlockB, false,
                      sparsityB.data());

  /////////////////// At * B1 = C1
  const int blockRowsB1 = blockRowsA;
  const int blockColsB1 = blockColsC;

  const int blockRowsC1 = blockColsA;
  const int blockColsC1 = blockColsB;

  const int rowsInBlockB1 = rowsInBlockA;
  const int colsInBlockB1 = colsInBlockC;

  const int rowsB1 = rowsInBlockB1 * blockRowsB1;
  const int colsB1 = colsInBlockB1 * blockColsB1;

  unsigned outBlocks1 = blockRowsC1 * blockColsC1;

  boost::multi_array<float, 2> hostB1;
  populateMatrixData1(hostB1, rowsB1, colsB1, rowsInBlockB1, colsInBlockB1,
                      // Swap rows and columns twice here
                      rowInBlockNzA, colInBlockNzA, blockRowNzA, blockColNzA);

  boost::multi_array<float, 2> hostAt = transpose(hostA);

  std::vector<unsigned char> sparsityC1(blockColsC1 * blockRowsC1, 1);

  // Mask [1][0] element out
  const int blockRowZero = 1;
  const int blockColZero = 0;
  sparsityC1[blockRowZero * blockColsB + blockColZero] = 0;
  --outBlocks1;

  // At[colsA x rowsA]
  BlockDenseMatrix At(colsA, rowsA, colsInBlockA, rowsInBlockA, false);

  BlockDenseMatrix B1(rowsB1, colsB1, rowsInBlockB1, colsInBlockB1, false);

  ///////////////////
  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> streamMaps;

  /////////////////// A * B = C
  poplar::Tensor tensorA = A.createTensor(graph, dataType, "A");
  A.setBlockTensor(tensorA);

  std::unique_ptr<char[]> rawHostA =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorA, "A", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostA, dataType, rawHostA.get());

  poplar::Tensor tensorB = B.createTensor(graph, dataType, "B");
  B.setBlockTensor(tensorB);

  std::unique_ptr<char[]> blocksRawHostB =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB, "B", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, blocksHostB, dataType, blocksRawHostB.get());

  // No reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, 1, 1);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  std::vector<int> tileAssignment(hg.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  std::size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == outBlocks);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;
  for (auto iter = tensorCParts.begin(); iter != tensorCParts.end(); ++iter) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }

  /////////////////// At * B1 = C1
  poplar::Tensor tensorB1 = B1.createTensor(graph, dataType, "B1");
  B1.setBlockTensor(tensorB1);

  std::unique_ptr<char[]> rawHostB1 =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorB1, "B1", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostB1, dataType, rawHostB1.get());

#if 0 // Reference path for tensor At

  poplar::Tensor tensorAt = At.createTensor(graph, dataType, "At");
  At.setBlockTensor(tensorAt);

  std::unique_ptr<char[]> rawHostAt =
      poplibs_test::util::allocateHostMemoryForTensor(
          tensorAt, "At", graph, uploadProg, downloadProg, streamMaps);
  poplibs_test::util::copy(target, hostAt, dataType, rawHostAt.get());

#else

  // Reusing existing tensor in transposed matrix
  poplar::Tensor tensorAt = tensorA.transpose();
  At.setBlockTensor(tensorAt);

#endif

  // No reduction
  HyperGraph hg1(At, B1, dataType, dataType, dataType, 1, 1);
  hg1.createGraphMatMulSparsifyResult(sparsityC1.data(), MEMORY_CYCLE_RATIO,
                                      graph, "C1");

  std::vector<int> tileAssignment1(hg1.getTotalNodes(), 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts1;
  std::vector<unsigned int> nodeCTileId1;
  poplar::program::Sequence matMulProg1;
  hg1.createComputeSetMatMul(tileAssignment1, tensorCParts1, nodeCTileId1,
                             graph, "matMul1", matMulProg1);
  std::size_t tensorCPartsLen1 = tensorCParts1.size();
  BOOST_TEST(tensorCPartsLen1 == outBlocks1);

  std::vector<std::unique_ptr<char[]>> rawHostC1Parts;
  for (auto iter = tensorCParts1.begin(); iter != tensorCParts1.end(); ++iter) {
    rawHostC1Parts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        iter->second, std::string("C1_") + std::to_string(iter->first), graph,
        uploadProg, downloadProg, streamMaps));
  }
  ///////////////////

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(matMulProg1);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  /////////////////// C
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (std::size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  std::vector<std::vector<float>> hostC = matMul(hostA, hostB);

  checkDenseResult(dataType, blockRowsC, blockColsC, rowsInBlockC, colsInBlockC,
                   hostC, blocksHostCParts);

  /////////////////// C1
  std::vector<boost::multi_array<float, 2>> blocksHostC1Parts;
  for (int br = 0, outBlockIdx = 0; br < blockRowsC1; ++br) {
    for (int bc = 0; bc < blockColsC1; ++bc) {
      boost::multi_array<float, 2> blocksHostC1Part(
          boost::extents[rowsInBlockC][colsInBlockC]);
      std::fill_n(blocksHostC1Part.data(), blocksHostC1Part.num_elements(),
                  0.0f);
      if (sparsityC1[br * blockColsC1 + bc] > 0) {
        poplibs_test::util::copy(target, dataType,
                                 rawHostC1Parts[outBlockIdx].get(),
                                 blocksHostC1Part);
        ++outBlockIdx;
      }
      blocksHostC1Parts.push_back(std::move(blocksHostC1Part));
    }
  }

  std::vector<std::vector<float>> hostC1 = matMul(hostAt, hostB1);

  checkSparseResult(dataType, blockRowsC1, blockColsC1, rowsInBlockC,
                    colsInBlockC, hostC1, blocksHostC1Parts, sparsityC1);
}
