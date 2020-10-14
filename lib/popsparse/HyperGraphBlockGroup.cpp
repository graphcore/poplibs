// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "HyperGraphBlockGroup.hpp"
#include <algorithm>
#include <cmath>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>
#include <stdlib.h>

#define DEBUG_INFO 0

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

HyperGraphBlockGroup::HyperGraphBlockGroup(BlockMatrix &A, BlockMatrix &B,
                                           poplar::Type inDataTypeIn,
                                           poplar::Type outDataTypeIn,
                                           poplar::Type partialDataTypeIn,
                                           int nTileIn)
    : HyperGraphBlock(A, B, inDataTypeIn, outDataTypeIn, partialDataTypeIn,
                      nTileIn, 0.0f),
      isResultSparse(B.isDense()), wA(0), wB(0), wV(0), P(1),
      doRowSplit(false) {

  const unsigned inDataTypeSize = (inDataType == poplar::FLOAT) ? 4 : 2;
  const unsigned outDataTypeSize = (outDataType == poplar::FLOAT) ? 4 : 2;
  const unsigned partialDataTypeSize =
      (partialDataType == poplar::FLOAT) ? 4 : 2;

  unsigned wConvVertex = 0;   // How to estimate it?
  unsigned wReduceVertex = 0; // How to estimate it?
  wA = matA.getBlockRow() * matA.getBlockCol() * inDataTypeSize;
  wB = matB.getBlockRow() * matB.getBlockCol() * inDataTypeSize;
  wV = wConvVertex +
       matA.getBlockRow() * matB.getBlockCol() * partialDataTypeSize;
  wP = matA.getBlockRow() * matB.getBlockCol() * partialDataTypeSize;
  wC =
      wReduceVertex + matA.getBlockRow() * matB.getBlockCol() * outDataTypeSize;

  unsigned nRowB = matB.getBlockRowCount();
  unsigned nColB = matB.getBlockColCount();
  nonZeroBlocksByCol.resize(nColB, 0);
  for (unsigned j = 0; j < nColB; j++) {
    for (unsigned k = 0; k < nRowB; k++) {
      if (blockIdMatrixB[k][j] != -1) {
        nonZeroBlocksByCol[j]++;
      }
    }
  }
  nonZeroBlocksTotal =
      std::accumulate(nonZeroBlocksByCol.begin(), nonZeroBlocksByCol.end(), 0);

  logging::popsparse::trace("Non-zero blocks by column:");
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    for (unsigned c = 0; c < nColB; c++) {
      printf("%3d ", nonZeroBlocksByCol[c]);
    }
    printf("\n");
  }

  estimateP();
  estimateTileW();
  if (wTileEstRowSplit < wTileEstColSplit) {
    doRowSplit = true;
    P = estPRowSplit[0];
  } else {
    P = estPColSplit;
  }

  logging::popsparse::debug("Estimated best P:");
  logging::popsparse::debug(
      "col split = {}, row split = {}, col split eq reduce = [{},{}], row "
      "split eq reduce = [{},{},{},{}]",
      estPColSplit, estPRowSplit[0], estPColSplitMmEqRd[0],
      estPColSplitMmEqRd[1], estPRowSplitMmEqRd[0], estPRowSplitMmEqRd[1],
      estPRowSplitMmEqRd[2], estPRowSplitMmEqRd[3]);
  logging::popsparse::debug("Estimated tile weight (kB):");
  logging::popsparse::debug("col split = {}, row split = {}",
                            wTileEstColSplit / KBf, wTileEstRowSplit / KBf);

  logging::popsparse::debug("Start with {} split, P = {}",
                            doRowSplit ? "row" : "column", P);
}

unsigned HyperGraphBlockGroup::getPartNum(unsigned n, unsigned k, unsigned m) {
  unsigned nModK = n % k;
  unsigned p = n / k;
  unsigned mDivPp1 = m / (p + 1);
  if (nModK <= mDivPp1) {
    auto x = m - nModK * (p + 1);
    auto d = x / p;
    return nModK + d;
  } else {
    return mDivPp1;
  }
}

void HyperGraphBlockGroup::fillNodeAB2VMapping() {
  for (std::size_t i = 0; i < edgeA.size(); ++i) {
    const auto &e = edgeA[i];
    assert(e.in.size() == 1);
    unsigned int idNodeA = e.in[0];
    for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
      unsigned int idNodeV = e.out[i_out];
      const auto &nv = nodeV.at(nodeVIdxById.at(idNodeV));
      nodeAPartitions[idNodeA].insert(nv.partition);
      nodeVInputMapping[idNodeV][idNodeA] = NodeType::A;
    }
  }
  for (std::size_t i = 0; i < edgeB.size(); ++i) {
    const auto &e = edgeB[i];
    assert(e.in.size() == 1);
    unsigned int idNodeB = e.in[0];
    for (std::size_t i_out = 0; i_out < e.out.size(); ++i_out) {
      unsigned int idNodeV = e.out[i_out];
      const auto &nv = nodeV.at(nodeVIdxById.at(idNodeV));
      nodeBPartitions[idNodeB].insert(nv.partition);
      nodeVInputMapping[idNodeV][idNodeB] = NodeType::B;
    }
  }
}

void HyperGraphBlockGroup::fillNodeC2VMapping() {
  for (std::size_t i = 0; i < edgeC.size(); ++i) {
    const auto &e = edgeC[i];
    assert(e.out.size() == 1);
    unsigned int idNodeC = e.out[0];
    for (std::size_t i_in = 0; i_in < e.in.size(); ++i_in) {
      unsigned int idNodeV = e.in[i_in];
      nodeCInputMapping[idNodeC].push_back(idNodeV);
    }
  }
}

HyperGraphBlockGroup::TaDbg::TaDbg(unsigned numPartitionsIn,
                                   unsigned blockRowsC, unsigned blockColsC,
                                   unsigned blockColsA)
    : numPartitions(std::min(numPartitionsIn, 8U)),
      numRowsC(std::min(blockRowsC, 64U)),
      numColsC(std::min(blockColsC, 64U / numPartitions)),
      numColsA(std::min(blockColsA, 64U)) {
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    v.resize(numRowsC, std::vector<std::vector<int>>(
                           numColsC, std::vector<int>(numPartitions, -1)));
    a.resize(numRowsC, std::vector<int>(numColsA, -1));
    b.resize(numColsA, std::vector<int>(numColsC, -1));
    c.resize(numRowsC, std::vector<int>(numColsC, -1));
  }
}

void HyperGraphBlockGroup::TaDbg::setA(unsigned blockRow, unsigned blockCol,
                                       int idTile) {
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    if (blockRow < numRowsC && blockCol < numColsA) {
      a[blockRow][blockCol] = idTile;
    }
  }
}

void HyperGraphBlockGroup::TaDbg::setB(unsigned blockRow, unsigned blockCol,
                                       int idTile) {
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    if (blockRow < numColsA && blockCol < numColsC) {
      b[blockRow][blockCol] = idTile;
    }
  }
}

void HyperGraphBlockGroup::TaDbg::setC(unsigned blockRow, unsigned blockCol,
                                       int idTile) {
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    if (blockRow < numRowsC && blockCol < numColsC) {
      c[blockRow][blockCol] = idTile;
    }
  }
}

void HyperGraphBlockGroup::TaDbg::setV(unsigned blockRow, unsigned blockCol,
                                       unsigned p, int idTile) {
  if (logging::popsparse::shouldLog(logging::Level::Trace)) {
    if (blockRow < numRowsC && blockCol < numColsC && p < numPartitions) {
      v[blockRow][blockCol][p] = idTile;
    }
  }
}

// Algorithm outline:
// Always put node A or B on a tile with the vertex that needs it for execution
// (if any) Try to minimize the amount of temporary copies in greedy fashion. If
// node A, B is unmapped to any vertex, put it on least occupied tile.
void HyperGraphBlockGroup::placeABNodes(
    const std::vector<std::vector<std::unordered_set<int>>> &tilesByCK,
    const std::vector<std::vector<std::unordered_set<int>>> &tilesByRK,
    std::vector<W> &tilesWeights, TaDbg &taDbg) {
  unsigned unmappedADbg = 0;
  unsigned unmappedBDbg = 0;

  for (std::size_t i = 0; i < nodeB.size(); ++i) {
    const auto &n = nodeB[i];
    std::vector<int> tilesToLand;
    auto iter = nodeBPartitions.find(n.id);
    if (iter != nodeBPartitions.end()) {
      const auto &partitionIndicies = iter->second;
      assert(partitionIndicies.size() == 1);
      for (auto iter = partitionIndicies.begin();
           iter != partitionIndicies.end(); ++iter) {
        unsigned p = *iter;
        const auto &tiles = tilesByCK[n.blockCol][p];
        tilesToLand.insert(tilesToLand.end(), tiles.begin(), tiles.end());
      }
    }
    int idTile;
    if (!tilesToLand.empty()) {
      // All those tiles must eventually contain eiter this node or
      // its copy. So, we choose the most occupied tile to decrease it's message
      // size.
      const auto &iterMax = std::max_element(
          tilesToLand.begin(), tilesToLand.end(),
          [&](const int &elem0, const int &elem1) {
            return tilesWeights[elem0].wTmp < tilesWeights[elem1].wTmp;
          });
      idTile = *iterMax;
      tilesWeights[idTile].wTmp -= wB;
    } else {
      /// Put unmapped node on a least occupied tile.
      auto iterMin = std::min_element(tilesWeights.begin(), tilesWeights.end(),
                                      [&](const W &elem0, const W &elem1) {
                                        return elem0.wTotal < elem1.wTotal;
                                      });
      idTile = iterMin - tilesWeights.begin();
      iterMin->wTotal += wB;
      ++unmappedBDbg;
    }
    tileAssignment[n.id] = idTile;
    taDbg.setB(n.blockRow, n.blockCol, idTile);
  }
  for (std::size_t i = 0; i < nodeA.size(); ++i) {
    const auto &n = nodeA[i];
    std::vector<int> tilesToLand;
    auto iter = nodeAPartitions.find(n.id);
    if (iter != nodeAPartitions.end()) {
      const auto &partitionIndicies = iter->second;
      for (auto iter = partitionIndicies.begin();
           iter != partitionIndicies.end(); ++iter) {
        unsigned p = *iter;
        const auto &tiles = tilesByRK[n.blockRow][p];
        tilesToLand.insert(tilesToLand.end(), tiles.begin(), tiles.end());
      }
    }
    int idTile;
    if (!tilesToLand.empty()) {
      // All those tiles must eventually contain eiter this node or
      // its copy. Total tile weight cannot change, but temporary size can.
      // We choose the most occupied tile to decrease it's temporary size.
      auto iterMax = std::max_element(tilesToLand.begin(), tilesToLand.end(),
                                      [&](const int &elem0, const int &elem1) {
                                        return tilesWeights[elem0].wTmp <
                                               tilesWeights[elem1].wTmp;
                                      });
      idTile = *iterMax;
      tilesWeights[idTile].wTmp -= wA;
    } else {
      // Put unmapped node on a least occupied tile.
      auto iterMin = std::min_element(tilesWeights.begin(), tilesWeights.end(),
                                      [&](const W &elem0, const W &elem1) {
                                        return elem0.wTotal < elem1.wTotal;
                                      });
      idTile = iterMin - tilesWeights.begin();
      iterMin->wTotal += wA;
      ++unmappedADbg;
    }
    tileAssignment[n.id] = idTile;
    taDbg.setA(n.blockRow, n.blockCol, idTile);
  }
}

// Algorithm outline:
// Always put node C on a tile where vertex exists that computes its partial.
// Try to minimize the maximum amount of memory on each tile in greedy
// fashion.
void HyperGraphBlockGroup::placeCNodes(std::vector<int> &tilesWeightsReduce,
                                       TaDbg &taDbg) {
  nodeCTileId.clear();
  for (std::size_t i = 0; i < nodeC.size(); ++i) {
    const auto &n = nodeC[i];
    std::vector<int> tilesWeightsReduceAdj = tilesWeightsReduce;
    // Subtract weights of child partials from corresponding tiles
    // before choosing the lightest tile,
    // because child partials will not be copied there
    auto iterInput = nodeCInputMapping.find(n.id);
    if (iterInput != nodeCInputMapping.end()) {
      const auto &nodeVIds = iterInput->second;
      for (unsigned idx = 0; idx < nodeVIds.size(); ++idx) {
        const auto &nv = nodeV.at(idx);
        int idTileV = tileAssignment.at(nv.id);
        tilesWeightsReduceAdj.at(idTileV) -= wP;
      }
    }
    int idTile;
    const auto &iterMin = std::min_element(tilesWeightsReduceAdj.begin(),
                                           tilesWeightsReduceAdj.end());
    idTile = iterMin - tilesWeightsReduceAdj.begin();
    nodeCTileId.push_back(idTile);

    // Adding node C weight
    tilesWeightsReduce[idTile] += wC;
    // Adding weights of all partials that needs to be copied
    if (iterInput != nodeCInputMapping.end()) {
      const auto &nodeVIds = iterInput->second;
      for (unsigned idx = 0; idx < nodeVIds.size(); ++idx) {
        const auto &nv = nodeV.at(idx);
        int idTileV = tileAssignment.at(nv.id);
        if (idTile != idTileV) {
          tilesWeightsReduce.at(idTile) += wP;
        }
      }
    }
    taDbg.setC(n.blockRow, n.blockCol, idTile);
  }
}

void HyperGraphBlockGroup::logTilesMapping(const TaDbg &taDbg) {
  if (!logging::popsparse::shouldLog(logging::Level::Trace)) {
    return;
  }
  printf("taDbg.v:\n");
  for (unsigned r = 0; r < taDbg.numRowsC; ++r) {
    for (unsigned c = 0; c < taDbg.numColsC; ++c) {
      for (unsigned p = 0; p < taDbg.numPartitions; ++p) {
        if (taDbg.v[r][c][p] >= 0) {
          printf("%4d ", taDbg.v[r][c][p]);
        } else {
          printf("     ");
        }
      }
      printf("|");
    }
    printf("\n");
  }
  printf("taDbg.b:\n");
  for (unsigned r = 0; r < taDbg.numColsA; ++r) {
    for (unsigned c = 0; c < taDbg.numColsC; ++c) {
      if (taDbg.b[r][c] >= 0) {
        printf("%4d ", taDbg.b[r][c]);
      } else {
        printf("     ");
      }
    }
    printf("\n");
  }
  printf("taDbg.a:\n");
  for (unsigned r = 0; r < taDbg.numRowsC; ++r) {
    for (unsigned c = 0; c < taDbg.numColsA; ++c) {
      if (taDbg.a[r][c] >= 0) {
        printf("%4d ", taDbg.a[r][c]);
      } else {
        printf("     ");
      }
    }
    printf("\n");
  }
  printf("taDbg.c:\n");
  for (unsigned r = 0; r < taDbg.numRowsC; ++r) {
    printf("%2d ", r);
    for (unsigned c = 0; c < taDbg.numColsC; ++c) {
      if (taDbg.c[r][c] >= 0) {
        printf("%4d ", taDbg.c[r][c]);
      } else {
        printf("     ");
      }
    }
    printf("\n");
  }
}

void HyperGraphBlockGroup::getCalcParams(float &M, float &N, float &K, float &T,
                                         float &Nz, float &Wv, float &Wa,
                                         float &Wb, float &Wc) {
  M = static_cast<float>(matA.getBlockRowCount());
  N = static_cast<float>(matB.getBlockColCount());
  K = static_cast<float>(matA.getBlockColCount());
  T = static_cast<float>(nTile);
  Nz = static_cast<float>(nonZeroBlocksTotal);
  Wv = static_cast<float>(wV);
  Wa = static_cast<float>(wA);
  Wb = static_cast<float>(wB);
  Wc = static_cast<float>(wC);
}

// Estimate the initial number of P
void HyperGraphBlockGroup::estimateP() {
  float kOptColSplit, kOptRowSplit0, kOptRowSplit1;
  float M, N, K, T, Nz, Wv, Wa, Wb, Wc;
  getCalcParams(M, N, K, T, Nz, Wv, Wa, Wb, Wc);
  if (!isResultSparse) {
    kOptColSplit = std::sqrt(Wb / Wv * Nz * T / M) / N;  // from Eq. (6),(29)
    kOptRowSplit0 = std::sqrt(Wa / Wv * Nz * T / M) / N; // from Eq. (12),(29)
    kOptRowSplit1 = std::sqrt(Wa / Wv * K * T / M / N);  // from Eq. (13)
  } else {
    kOptColSplit = std::sqrt(Wb / Wv * K * T / Nz); // from Eq. (17),(30)
    kOptRowSplit0 = kOptRowSplit1 =
        std::sqrt(Wa / Wv * K * T / Nz); // from Eq. (19),(30)
  }
  float kMax = Nz / N;
  kOptColSplit = std::max(1.0f, std::min(kOptColSplit, kMax));
  kOptRowSplit0 = std::max(1.0f, std::min(kOptRowSplit0, kMax));
  kOptRowSplit1 = std::max(1.0f, std::min(kOptRowSplit1, kMax));

  estPColSplit = std::max(1U, static_cast<unsigned>(std::round(kOptColSplit)));
  // 2 boundary cases
  estPRowSplit[0] =
      std::max(1U, static_cast<unsigned>(std::round(kOptRowSplit0)));
  estPRowSplit[1] =
      std::max(1U, static_cast<unsigned>(std::round(kOptRowSplit1)));

  if (!isResultSparse) {
    // DSD case
    if (M * N / T < 1.0f) {
      // Calculating matmul-reduce intersection for column split
      float a = Wv * (M * N / T - 1.0f); // Eq. (25.1)
      float b = Wa * M * Nz / T - Wc;    // From Eq. (25.2),(29)
      float c = Wb * Nz / N;             // From Eq. (25.3),(29)
      float dis = b * b - 4.0f * a * c;
      if (dis >= 0.0f) {
        float sqrtDis = std::sqrt(dis);
        float x1 = (-b + sqrtDis) * 0.5f / a;
        float x2 = (-b - sqrtDis) * 0.5f / a;
        if (x1 > 0.0f) {
          estPColSplitMmEqRd[0] = std::max(1, static_cast<int>(std::round(x1)));
        }
        if (x2 > 0.0f) {
          estPColSplitMmEqRd[1] = std::max(1, static_cast<int>(std::round(x2)));
        }
      }
      // Calculating matmul-reduce intersection for row split
      a = Wv * (M * N / T - 1.0f); // Eq. (26.1)
      b = Wb * M * Nz / T - Wc;    // From Eq. (26.2),(29)
      std::array<float, 2> c2;
      c2[0] = Wb * Nz / N; // From Eq. (26.3),(29)
      c2[1] = Wb * K;      // Eq. (26.3)
      std::array<float, 2> dis2;
      for (int i = 0, idx = 0; i < 2; ++i) {
        dis2[i] = b * b - 4.0f * a * c2[i];
        if (dis2[i] >= 0.0f) {
          float sqrtDis = std::sqrt(dis2[i]);
          float x1 = (-b + sqrtDis) * 0.5f / a;
          float x2 = (-b - sqrtDis) * 0.5f / a;
          if (x1 > 0.0f) {
            estPRowSplitMmEqRd[idx++] =
                std::max(1, static_cast<int>(std::round(x1)));
          }
          if (x2 > 0.0f) {
            estPRowSplitMmEqRd[idx++] =
                std::max(1, static_cast<int>(std::round(x2)));
          }
        }
      }
    }
  } else {
    // DDS case
    if (Nz < T) {
      // Calculating matmul-reduce intersection for column split
      float a = Wv * (Nz / T - 1.0f); // From Eq. (27.1),(30)
      float b = Wa * Nz * K / T - Wc; // From Eq. (27.2),(30)
      float c = Wb * K;               // Eq. (27.3)
      float dis = b * b - 4.0f * a * c;
      if (dis >= 0.0f) {
        float sqrtDis = std::sqrt(dis);
        float x1 = (-b + sqrtDis) * 0.5f / a;
        float x2 = (-b - sqrtDis) * 0.5f / a;
        if (x1 > 0.0f) {
          estPColSplitMmEqRd[0] = std::max(1, static_cast<int>(std::round(x1)));
        }
        if (x2 > 0.0f) {
          estPColSplitMmEqRd[1] = std::max(1, static_cast<int>(std::round(x2)));
        }
      }
      // Calculating matmul-reduce intersection for row split
      // a is the same
      b = Wb * Nz * K / T - Wc; // From Eq. (28.2),(30)
      c = Wa * K;               // Eq. (28.3)
      dis = b * b - 4.0f * a * c;
      if (dis >= 0.0f) {
        float sqrtDis = std::sqrt(dis);
        float x1 = (-b + sqrtDis) * 0.5f / a;
        float x2 = (-b - sqrtDis) * 0.5f / a;
        if (x1 > 0.0f) {
          estPRowSplitMmEqRd[0] = std::max(1, static_cast<int>(std::round(x1)));
        }
        if (x2 > 0.0f) {
          estPRowSplitMmEqRd[1] = std::max(1, static_cast<int>(std::round(x2)));
        }
      }
    }
  }
}

// Get the estimated weight of matmul for DSD column split. From Eq. (5),(29)
static float getWMmDSDColSplit(float M, float N, float K, float T, float P,
                               float Nz, float Wv, float Wa, float Wb,
                               float Wc) {
  return Wv * M * N * P / T + Wa * M * Nz / T + Wb * Nz / P / N;
}

// Get the estimated weight of matmul for DSD row split. From Eq. (11),(29)
static float getWMmDSDRowSplit(float M, float N, float K, float T, float P,
                               float Nz, float Wv, float Wa, float Wb,
                               float Wc) {
  float S = Nz / (K * N);
  float beta =
      1.0f - std::pow(1.0f - S, std::max(1.0f, std::round(M * N * P / T)));
  return Wv * M * N * P / T + Wb * M * Nz / T + Wa * beta * K / P;
}

// Get the estimated weight of reduce for DSD. Eq. (22)
static float getWRdDSD(float M, float N, float K, float T, float P, float Nz,
                       float Wv, float Wa, float Wb, float Wc) {
  return std::max(1.0f, M * N / T) * (Wc + Wv * P);
}

// Get the estimated weight of matmul for DDS column split. From Eq. (16),(30)
static float getWMmDDSColSplit(float M, float N, float K, float T, float P,
                               float Nz, float Wv, float Wa, float Wb,
                               float Wc) {
  return Wv * P * Nz / T + Wa * K * Nz / T + Wb * K / P;
}

// Get the estimated weight of matmul for DDS row split. From Eq. (18),(30)
static float getWMmDDSRowSplit(float M, float N, float K, float T, float P,
                               float Nz, float Wv, float Wa, float Wb,
                               float Wc) {
  return Wv * P * Nz / T + Wb * K * Nz / T + Wa * K / P;
}

// Get the estimated weight of reduce for DDS. From Eq. (24),(30)
static float getWRdDDS(float M, float N, float K, float T, float P, float Nz,
                       float Wv, float Wa, float Wb, float Wc) {
  return std::max(1.0f, Nz / T) * (Wc + Wv * P);
}

void HyperGraphBlockGroup::estimateTileW() {
  float M, N, K, T, Nz, Wv, Wa, Wb, Wc;
  getCalcParams(M, N, K, T, Nz, Wv, Wa, Wb, Wc);
  float P;
  if (!isResultSparse) {
    // DSD case
    // column split
    // Take points where matmul is minimum and estimate matmul and reduce
    // weights
    P = static_cast<float>(estPColSplit);
    float wTileEstColSplitMm =
        getWMmDSDColSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    float wTileEstColSplitRd = getWRdDSD(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstColSplit = std::max(wTileEstColSplitMm, wTileEstColSplitRd);
    // Take points where matmul is equal to reduce and estimate matmul and
    // reduce weights
    for (int i = 0; i < 2; ++i) {
      if (estPColSplitMmEqRd[i] > 0) {
        P = static_cast<float>(estPColSplitMmEqRd[i]);
        wTileEstColSplitMm =
            getWMmDSDColSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        wTileEstColSplitRd = getWRdDSD(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        float wTileEstColSplitAtEqRd =
            std::max(wTileEstColSplitMm, wTileEstColSplitRd);
        if (wTileEstColSplitAtEqRd < wTileEstColSplit) {
          wTileEstColSplit = wTileEstColSplitAtEqRd;
          estPColSplit = estPColSplitMmEqRd[i];
        }
      }
    }
    // row split
    std::array<float, 2> wTileEstRowSplitMax;
    // Take points where matmul is minimum and estimate matmul and reduce
    // weights
    // First approximate point
    P = static_cast<float>(estPRowSplit[0]);
    float wTileEstRowSplitMm =
        getWMmDSDRowSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    float wTileEstRowSplitRd = getWRdDSD(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstRowSplitMax[0] = std::max(wTileEstRowSplitMm, wTileEstRowSplitRd);
    // Second approximate point
    P = static_cast<float>(estPRowSplit[1]);
    wTileEstRowSplitMm = getWMmDSDRowSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstRowSplitRd = getWRdDSD(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstRowSplitMax[1] = std::max(wTileEstRowSplitMm, wTileEstRowSplitRd);
    if (wTileEstRowSplitMax[0] < wTileEstRowSplitMax[1]) {
      wTileEstRowSplit = wTileEstRowSplitMax[0];
      estPRowSplit[1] = -1;
    } else {
      wTileEstRowSplit = wTileEstRowSplitMax[1];
      estPRowSplit[0] = estPRowSplit[1];
      estPRowSplit[1] = -1;
    }
    // Take points where matmul is equal to reduce and estimate matmul and
    // reduce weights
    for (int i = 0; i < 4; ++i) {
      if (estPRowSplitMmEqRd[i] > 0) {
        P = static_cast<float>(estPRowSplitMmEqRd[i]);
        wTileEstRowSplitMm =
            getWMmDSDRowSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        wTileEstRowSplitRd = getWRdDSD(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        float wTileEstRowSplitAtEqRd =
            std::max(wTileEstRowSplitMm, wTileEstRowSplitRd);
        if (wTileEstRowSplitAtEqRd < wTileEstRowSplit) {
          wTileEstRowSplit = wTileEstRowSplitAtEqRd;
          estPRowSplit[0] = estPRowSplitMmEqRd[i];
        }
      }
    }
  } else {
    // DDS case
    // column split
    // Take points where matmul is minimum and estimate matmul and reduce
    // weights
    P = static_cast<float>(estPColSplit);
    float wTileEstColSplitMm =
        getWMmDDSColSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    float wTileEstColSplitRd = getWRdDDS(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstColSplit = std::max(wTileEstColSplitMm, wTileEstColSplitRd);
    // Take points where matmul is equal to reduce and estimate matmul and
    // reduce weights
    for (int i = 0; i < 2; ++i) {
      if (estPColSplitMmEqRd[i] > 0) {
        P = static_cast<float>(estPColSplitMmEqRd[i]);
        wTileEstColSplitMm =
            getWMmDDSColSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        wTileEstColSplitRd = getWRdDDS(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        float wTileEstColSplitAtEqRd =
            std::max(wTileEstColSplitMm, wTileEstColSplitRd);
        if (wTileEstColSplitAtEqRd < wTileEstColSplit) {
          wTileEstColSplit = wTileEstColSplitAtEqRd;
          estPColSplit = estPColSplitMmEqRd[i];
        }
      }
    }
    // row split
    // Take points where matmul is minimum and estimate matmul and reduce
    // weights
    P = static_cast<float>(estPRowSplit[0]);
    float wTileEstRowSplitMm =
        getWMmDDSRowSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    float wTileEstRowSplitRd = getWRdDDS(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
    wTileEstRowSplit = std::max(wTileEstRowSplitMm, wTileEstRowSplitRd);
    // Take points where matmul is equal to reduce and estimate matmul and
    // reduce weights
    for (int i = 0; i < 2; ++i) {
      if (estPRowSplitMmEqRd[i] > 0) {
        P = static_cast<float>(estPRowSplitMmEqRd[i]);
        wTileEstRowSplitMm =
            getWMmDDSRowSplit(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        wTileEstRowSplitRd = getWRdDDS(M, N, K, T, P, Nz, Wv, Wa, Wb, Wc);
        float wTileEstRowSplitAtEqRd =
            std::max(wTileEstRowSplitMm, wTileEstRowSplitRd);
        if (wTileEstRowSplitAtEqRd < wTileEstRowSplit) {
          wTileEstRowSplit = wTileEstRowSplitAtEqRd;
          estPRowSplit[0] = estPRowSplitMmEqRd[i];
        }
      }
    }
  }
}

} // namespace experimental
} // namespace popsparse