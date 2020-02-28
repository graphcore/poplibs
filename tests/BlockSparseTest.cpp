// Copyright (c) Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE BlockSparseTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_test/Util.hpp>
#include <stdlib.h>
#include <vector>

#include "popsparse/BSMatrix.hpp"
#include "popsparse/HyperGraph.hpp"
#include "popsparse/experimental/BlockSparseMatMul.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popsparse::experimental;

static const float MEMORY_CYCLE_RATIO = 1.0f;

// TODO: Move to some kind of utils
void getSparseMatrixBlocks(
    int rows, int cols, int rowsInBlock, int colsInBlock,
    std::vector<unsigned char> &sparsity,
    const boost::multi_array<float, 2> &denseMat,
    std::vector<boost::multi_array<float, 1>> &blockData) {
  const int blockRows = rows / rowsInBlock;
  const int blockCols = cols / colsInBlock;

  int blockCount = 0;
  for (int br = 0; br < blockRows; br++) {
    for (int bc = 0; bc < blockCols; bc++) {
      if (sparsity[br * blockCols + bc] == 1) {
        blockData.push_back(boost::multi_array<float, 1>(
            boost::extents[rowsInBlock * colsInBlock]));
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

// TODO: Move to some kind of utils
void populateMatrixData1(boost::multi_array<float, 2> &hostA,
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

  // We make only one element of a block as non-zero for simplicity
  // hostA is row-major
  for (int r = 0; r < rowsA; ++r) {
    int rb = r % rowsInBlockA;
    int br = r / rowsInBlockA;
    for (int c = 0; c < colsA; ++c) {
      int cb = c % colsInBlockA;
      int bc = c / colsInBlockA;
      if (rb == rowInBlockNzA && cb == colInBlockNzA &&
          (blockRowNzA < 0 || br == blockRowNzA) &&
          (blockColNzA < 0 || bc == blockColNzA)) {
        hostA[r][c] = 1.0f;
      } else {
        hostA[r][c] = 0.0f;
      }
    }
  }

  // hostB is row-major
  for (int r = 0; r < rowsB; ++r) {
    int rb = r % rowsInBlockB;
    int br = r / rowsInBlockB;
    for (int c = 0; c < colsB; ++c) {
      int cb = c % colsInBlockB;
      int bc = c / colsInBlockB;
      if (rb == rowInBlockNzB && cb == colInBlockNzB &&
          (blockRowNzB < 0 || br == blockRowNzB) &&
          (blockColNzB < 0 || bc == blockColNzB)) {
        hostB[r][c] = 1.0f;
      } else {
        hostB[r][c] = 0.0f;
      }
    }
  }
}

void populateMatrixData1(const std::array<int, 3> &dim,
                         const std::array<int, 3> &block_size,
                         boost::multi_array<float, 2> &lhsMatrix,
                         boost::multi_array<float, 2> &rhsMatrix,
                         const std::array<int, 2> &nzBlockElem,
                         const std::array<int, 2> &nzBlock) {
  populateMatrixData1(lhsMatrix, rhsMatrix, dim[0], dim[1], dim[2],
                      block_size[0], block_size[1], block_size[3],
                      nzBlockElem[0], nzBlockElem[1], nzBlock[0], nzBlock[1]);
}

/*
Computes matrix multiplication.
All matrices are row-major
*/
std::vector<std::vector<float>>
matMul(const boost::multi_array<float, 2> &hostA,
       const boost::multi_array<float, 2> &hostB, int rowsA, int colsA,
       int colsB) {
  std::vector<std::vector<float>> hostC;
  for (int r = 0; r < rowsA; ++r) {
    hostC.push_back({});
    for (int c = 0; c < colsB; ++c) {
      float sum = 0.0f;
      for (int mulc = 0; mulc < colsA; ++mulc) {
        sum += hostA[r][mulc] * hostB[mulc][c];
      }
      hostC[r].push_back(sum);
    }
  }
  return hostC;
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

  BOOST_TEST(nodeA.size() == 4);
  BOOST_TEST(nodeB.size() == 4);
  BOOST_TEST(nodeC.size() == 4);
  BOOST_TEST(nodeV.size() == 4);

  BOOST_TEST(edgeA.size() == 4);
  BOOST_TEST(edgeB.size() == 4);
  BOOST_TEST(edgeC.size() == 4);

  BOOST_TEST(nodeV[0].idxA.size() == 2);
  BOOST_TEST(nodeV[0].idxB.size() == 2);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[1].idxA.size() == 2);
  BOOST_TEST(nodeV[1].idxB.size() == 2);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(nodeV[2].idxA.size() == 2);
  BOOST_TEST(nodeV[2].idxB.size() == 2);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[3].idxA.size() == 2);
  BOOST_TEST(nodeV[3].idxB.size() == 2);
  BOOST_TEST(nodeV[3].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[3].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(edgeA[0].in.size() == 1);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[1].in.size() == 1);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[2].in.size() == 1);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 2);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeA[3].in.size() == 1);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 2);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeB[0].in.size() == 1);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 2);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeB[1].in.size() == 1);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeB[2].in.size() == 1);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 2);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeB[3].in.size() == 1);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeC[0].out.size() == 1);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 1);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeC[1].out.size() == 1);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 1);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[1].id}));

  BOOST_TEST(edgeC[2].out.size() == 1);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 1);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeC[3].out.size() == 1);
  BOOST_TEST(edgeC[3].out == std::vector<unsigned>({nodeC[3].id}));
  BOOST_TEST(edgeC[3].in.size() == 1);
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

  BOOST_TEST(nodeA.size() == 4);
  BOOST_TEST(nodeB.size() == 4);
  BOOST_TEST(nodeC.size() == 4);
  BOOST_TEST(nodeV.size() == 8);

  BOOST_TEST(edgeA.size() == 4);
  BOOST_TEST(edgeB.size() == 4);
  BOOST_TEST(edgeC.size() == 4);

  BOOST_TEST(nodeV[0].idxA.size() == 1);
  BOOST_TEST(nodeV[0].idxB.size() == 1);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0}));

  BOOST_TEST(nodeV[1].idxA.size() == 1);
  BOOST_TEST(nodeV[1].idxB.size() == 1);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({2}));

  BOOST_TEST(nodeV[2].idxA.size() == 1);
  BOOST_TEST(nodeV[2].idxB.size() == 1);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({0}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({1}));

  BOOST_TEST(nodeV[3].idxA.size() == 1);
  BOOST_TEST(nodeV[3].idxB.size() == 1);
  BOOST_TEST(nodeV[3].idxA == std::vector<unsigned>({1}));
  BOOST_TEST(nodeV[3].idxB == std::vector<unsigned>({3}));

  BOOST_TEST(nodeV[4].idxA.size() == 1);
  BOOST_TEST(nodeV[4].idxB.size() == 1);
  BOOST_TEST(nodeV[4].idxA == std::vector<unsigned>({2}));
  BOOST_TEST(nodeV[4].idxB == std::vector<unsigned>({0}));

  BOOST_TEST(nodeV[5].idxA.size() == 1);
  BOOST_TEST(nodeV[5].idxB.size() == 1);
  BOOST_TEST(nodeV[5].idxA == std::vector<unsigned>({3}));
  BOOST_TEST(nodeV[5].idxB == std::vector<unsigned>({2}));

  BOOST_TEST(nodeV[6].idxA.size() == 1);
  BOOST_TEST(nodeV[6].idxB.size() == 1);
  BOOST_TEST(nodeV[6].idxA == std::vector<unsigned>({2}));
  BOOST_TEST(nodeV[6].idxB == std::vector<unsigned>({1}));

  BOOST_TEST(nodeV[7].idxA.size() == 1);
  BOOST_TEST(nodeV[7].idxB.size() == 1);
  BOOST_TEST(nodeV[7].idxA == std::vector<unsigned>({3}));
  BOOST_TEST(nodeV[7].idxB == std::vector<unsigned>({3}));

  BOOST_TEST(edgeA[0].in.size() == 1);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[2].id}));

  BOOST_TEST(edgeA[1].in.size() == 1);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[3].id}));

  BOOST_TEST(edgeA[2].in.size() == 1);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 2);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[4].id, nodeV[6].id}));

  BOOST_TEST(edgeA[3].in.size() == 1);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 2);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[5].id, nodeV[7].id}));

  BOOST_TEST(edgeB[0].in.size() == 1);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 2);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[4].id}));

  BOOST_TEST(edgeB[1].in.size() == 1);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[2].id, nodeV[6].id}));

  BOOST_TEST(edgeB[2].in.size() == 1);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 2);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[1].id, nodeV[5].id}));

  BOOST_TEST(edgeB[3].in.size() == 1);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[3].id, nodeV[7].id}));

  BOOST_TEST(edgeC[0].out.size() == 1);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 2);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeC[1].out.size() == 1);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 2);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[2].id, nodeV[3].id}));

  BOOST_TEST(edgeC[2].out.size() == 1);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 2);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[4].id, nodeV[5].id}));

  BOOST_TEST(edgeC[3].out.size() == 1);
  BOOST_TEST(edgeC[3].out == std::vector<unsigned>({nodeC[3].id}));
  BOOST_TEST(edgeC[3].in.size() == 2);
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
BOOST_AUTO_TEST_CASE(HyperGraph_testMatMulOuterNoReduction) {
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

  BOOST_TEST(nodeA.size() == 4);
  BOOST_TEST(nodeB.size() == 4);
  BOOST_TEST(nodeC.size() == 3);
  BOOST_TEST(nodeV.size() == 3);

  BOOST_TEST(edgeA.size() == 4);
  BOOST_TEST(edgeB.size() == 4);
  BOOST_TEST(edgeC.size() == 3);

  BOOST_TEST(nodeV[0].idxA.size() == 2);
  BOOST_TEST(nodeV[0].idxB.size() == 2);
  BOOST_TEST(nodeV[0].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[0].idxB == std::vector<unsigned>({0, 2}));

  BOOST_TEST(nodeV[1].idxA.size() == 2);
  BOOST_TEST(nodeV[1].idxB.size() == 2);
  BOOST_TEST(nodeV[1].idxA == std::vector<unsigned>({0, 1}));
  BOOST_TEST(nodeV[1].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(nodeV[2].idxA.size() == 2);
  BOOST_TEST(nodeV[2].idxB.size() == 2);
  BOOST_TEST(nodeV[2].idxA == std::vector<unsigned>({2, 3}));
  BOOST_TEST(nodeV[2].idxB == std::vector<unsigned>({1, 3}));

  BOOST_TEST(edgeA[0].in.size() == 1);
  BOOST_TEST(edgeA[0].in == std::vector<unsigned>({nodeA[0].id}));
  BOOST_TEST(edgeA[0].out.size() == 2);
  BOOST_TEST(edgeA[0].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[1].in.size() == 1);
  BOOST_TEST(edgeA[1].in == std::vector<unsigned>({nodeA[1].id}));
  BOOST_TEST(edgeA[1].out.size() == 2);
  BOOST_TEST(edgeA[1].out == std::vector<unsigned>({nodeV[0].id, nodeV[1].id}));

  BOOST_TEST(edgeA[2].in.size() == 1);
  BOOST_TEST(edgeA[2].in == std::vector<unsigned>({nodeA[2].id}));
  BOOST_TEST(edgeA[2].out.size() == 1);
  BOOST_TEST(edgeA[2].out == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeA[3].in.size() == 1);
  BOOST_TEST(edgeA[3].in == std::vector<unsigned>({nodeA[3].id}));
  BOOST_TEST(edgeA[3].out.size() == 1);
  BOOST_TEST(edgeA[3].out == std::vector<unsigned>({nodeV[2].id}));

  BOOST_TEST(edgeB[0].in.size() == 1);
  BOOST_TEST(edgeB[0].in == std::vector<unsigned>({nodeB[0].id}));
  BOOST_TEST(edgeB[0].out.size() == 1);
  BOOST_TEST(edgeB[0].out == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeB[1].in.size() == 1);
  BOOST_TEST(edgeB[1].in == std::vector<unsigned>({nodeB[1].id}));
  BOOST_TEST(edgeB[1].out.size() == 2);
  BOOST_TEST(edgeB[1].out == std::vector<unsigned>({nodeV[1].id, nodeV[2].id}));

  BOOST_TEST(edgeB[2].in.size() == 1);
  BOOST_TEST(edgeB[2].in == std::vector<unsigned>({nodeB[2].id}));
  BOOST_TEST(edgeB[2].out.size() == 1);
  BOOST_TEST(edgeB[2].out == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeB[3].in.size() == 1);
  BOOST_TEST(edgeB[3].in == std::vector<unsigned>({nodeB[3].id}));
  BOOST_TEST(edgeB[3].out.size() == 2);
  BOOST_TEST(edgeB[3].out == std::vector<unsigned>({nodeV[1].id, nodeV[2].id}));

  BOOST_TEST(edgeC[0].out.size() == 1);
  BOOST_TEST(edgeC[0].out == std::vector<unsigned>({nodeC[0].id}));
  BOOST_TEST(edgeC[0].in.size() == 1);
  BOOST_TEST(edgeC[0].in == std::vector<unsigned>({nodeV[0].id}));

  BOOST_TEST(edgeC[1].out.size() == 1);
  BOOST_TEST(edgeC[1].out == std::vector<unsigned>({nodeC[1].id}));
  BOOST_TEST(edgeC[1].in.size() == 1);
  BOOST_TEST(edgeC[1].in == std::vector<unsigned>({nodeV[1].id}));

  BOOST_TEST(edgeC[2].out.size() == 1);
  BOOST_TEST(edgeC[2].out == std::vector<unsigned>({nodeC[2].id}));
  BOOST_TEST(edgeC[2].in.size() == 1);
  BOOST_TEST(edgeC[2].in == std::vector<unsigned>({nodeV[2].id}));
}

void TestMatMul(const poplar::Type &dataType, int blockSize, int blockRowsA,
                int blockColsA, int blockColsB, int rowInBlockNzA = 0,
                int colInBlockNzA = 0, int blockRowNzA = -1,
                int blockColNzA = -1);

/*
Testing assumetric layout of blocks and assymetric layout within a block
*/
BOOST_AUTO_TEST_CASE(MatMul_testF32_assymetric) {
  TestMatMul(FLOAT, 8, 2, 2, 2, 1, 2, 0, 1);
}

/*
Testing all combinations or number of block rows and columns from 1 to 2
*/
BOOST_AUTO_TEST_CASE(MatMul_testF32) {
  for (int blockRowsA = 1; blockRowsA <= 2; ++blockRowsA) {
    for (int blockColsA = 1; blockColsA <= 2; ++blockColsA) {
      for (int blockColsB = 1; blockColsB <= 2; ++blockColsB) {
        TestMatMul(FLOAT, 8, blockRowsA, blockColsA, blockColsB, 1, 2);
      }
    }
  }
}

/*
Testing fp16
*/
BOOST_AUTO_TEST_CASE(MatMul_testF16) { TestMatMul(HALF, 16, 1, 2, 1); }

/*
Testing MatMul vertex - no reduction case
*/
void TestMatMul(const poplar::Type &dataType, int blockSize, int blockRowsA,
                int blockColsA, int blockColsB, int rowInBlockNzA,
                int colInBlockNzA, int blockRowNzA, int blockColNzA) {

  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const int batchSize = 6;

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
  populateMatrixData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                      colsInBlockA, colsInBlockB, rowInBlockNzA, colInBlockNzA,
                      blockRowNzA, blockColNzA);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> hostBTransposed = transpose(hostB);

  // Blocks layout in blocksHostB are row-major
  // Individual blocks are column-major
  std::vector<boost::multi_array<float, 1>> blocksHostB;
  getSparseMatrixBlocks(colsB, rowsB, rowsInBlockB, colsInBlockB, sparsityB,
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

  std::vector<std::unique_ptr<char[]>> blocksRawHostB(tensorB.dim(0));
  for (unsigned int i = 0; i < tensorB.dim(0); i++) {
    blocksRawHostB[i] = poplibs_test::util::allocateHostMemoryForTensor(
        tensorB[i], "B_block_" + std::to_string(i), graph, uploadProg,
        downloadProg, streamMaps);
    poplibs_test::util::copy(target, blocksHostB[i], dataType,
                             blocksRawHostB[i].get());
  }

  // No reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, 1, 1);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  std::vector<int> partition;

  unsigned int nPartition = hg.getTotalNodes();
  // Put everything on tile 0
  std::vector<int> tileAssignment(nPartition, 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  size_t tensorCPartsLen = tensorCParts.size();
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

  const OptionFlags engineOptions{{"target.workerStackSizeInBytes", "0x200"},
                                  {"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  // hostC is row-major
  std::vector<std::vector<float>> hostC =
      matMul(hostA, hostB, rowsA, colsA, colsB);

  const float epsilon = 0.001f;
  int r = 0;
  for (int br = 0; br < blockRowsC; ++br) {
    for (int rb = 0; rb < rowsInBlockC; ++rb, ++r) {
      int c = 0;
      for (int bc = 0; bc < blockColsC; ++bc) {
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

void TestMatMulReduce(const poplar::Type &dataType);

BOOST_AUTO_TEST_CASE(MatMulReduce_testF32) { TestMatMulReduce(FLOAT); }

/*
Testing MatMul and Reduce vertices
*/
void TestMatMulReduce(const poplar::Type &dataType) {
  IPUModel ipuModel;
  auto device =
      createTestDevice(TEST_TARGET, ipuModel.numIPUs, ipuModel.tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  const int blockSize = 8;
  const int batchSize = 6;

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
  populateMatrixData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
                      colsInBlockA, colsInBlockB, 1, 2);

  std::vector<unsigned char> sparsityB(blockRowsB * blockColsB, 1);

  boost::multi_array<float, 2> hostBTransposed = transpose(hostB);

  std::vector<boost::multi_array<float, 1>> blocksHostB;
  getSparseMatrixBlocks(colsB, rowsB, rowsInBlockB, colsInBlockB, sparsityB,
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

  std::vector<std::unique_ptr<char[]>> blocksRawHostB(tensorB.dim(0));
  for (unsigned int i = 0; i < tensorB.dim(0); i++) {
    blocksRawHostB[i] = poplibs_test::util::allocateHostMemoryForTensor(
        tensorB[i], "B_block_" + std::to_string(i), graph, uploadProg,
        downloadProg, streamMaps);
    poplibs_test::util::copy(target, blocksHostB[i], dataType,
                             blocksRawHostB[i].get());
  }

  // Reduction
  HyperGraph hg(A, B, dataType, dataType, dataType, ipuModel.tilesPerIPU);
  hg.createGraphMatMul(MEMORY_CYCLE_RATIO, graph, "C");

  std::vector<int> partition;

  unsigned int nPartition = hg.getTotalNodes();
  // Put everything on tile 0
  std::vector<int> tileAssignment(nPartition, 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  size_t tensorCPartsLen = tensorCParts.size();
  BOOST_TEST(tensorCPartsLen == hg.getNodeV().size());

  poplar::program::Sequence reduceProg;
  hg.createComputeSetReduce(tensorCParts, nodeCTileId, graph, "test",
                            reduceProg);

  std::vector<std::unique_ptr<char[]>> rawHostCParts;

  for (size_t i = 0; i < hg.matC->getBlockTensor().size(); ++i) {
    rawHostCParts.push_back(poplibs_test::util::allocateHostMemoryForTensor(
        hg.matC->getBlockTensor()[i], std::string("C_") + std::to_string(i),
        graph, uploadProg, downloadProg, streamMaps));
  }

  Sequence allSequence;
  allSequence.add(uploadProg);
  allSequence.add(matMulProg);
  allSequence.add(reduceProg);
  allSequence.add(downloadProg);

  const OptionFlags engineOptions{{"target.workerStackSizeInBytes", "0x200"},
                                  {"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  for (size_t i = 0; i < rawHostCParts.size(); ++i) {
    boost::multi_array<float, 2> blocksHostCPart(
        boost::extents[rowsInBlockC][colsInBlockC]);
    poplibs_test::util::copy(target, dataType, rawHostCParts[i].get(),
                             blocksHostCPart);
    blocksHostCParts.push_back(std::move(blocksHostCPart));
  }

  std::vector<std::vector<float>> hostC =
      matMul(hostA, hostB, rowsA, colsA, colsB);

  const float epsilon = 0.001f;
  int r = 0;
  for (int br = 0; br < blockRowsC; ++br) {
    for (int rb = 0; rb < rowsInBlockC; ++rb, ++r) {
      int c = 0;
      for (int bc = 0; bc < blockColsC; ++bc) {
        int outBlockIdx = bc * blockRowsC + br;
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
  populateMatrixData1(hostA, hostB, rowsA, colsA, colsB, rowsInBlockA,
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

  std::vector<int> partition;

  unsigned int nPartition = hg.getTotalNodes();
  // Put everything on tile 0
  std::vector<int> tileAssignment(nPartition, 0);

  std::map<unsigned int, poplar::Tensor> tensorCParts;
  std::vector<unsigned int> nodeCTileId;
  poplar::program::Sequence matMulProg;
  hg.createComputeSetMatMul(tileAssignment, tensorCParts, nodeCTileId, graph,
                            "test", matMulProg);
  size_t tensorCPartsLen = tensorCParts.size();
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

  const OptionFlags engineOptions{{"target.workerStackSizeInBytes", "0x200"},
                                  {"debug.allowOutOfMemory", "true"}};

  Engine engine(graph, allSequence, engineOptions);
  poplibs_test::util::attachStreams(engine, streamMaps);
  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);
  });

  // The result is row-major
  std::vector<boost::multi_array<float, 2>> blocksHostCParts;
  size_t outBlockIdx = 0;
  for (size_t br = 0; br < blockRowsC; ++br) {
    for (size_t bc = 0; bc < blockColsC; ++bc) {
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
      ;
    }
  }

  std::vector<std::vector<float>> hostC =
      matMul(hostA, hostB, rowsA, colsA, colsB);

  const float epsilon = 0.001f;
  int r = 0;
  for (int br = 0; br < blockRowsC; ++br) {
    for (int rb = 0; rb < rowsInBlockC; ++rb, ++r) {
      int c = 0;
      for (int bc = 0; bc < blockColsC; ++bc) {
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