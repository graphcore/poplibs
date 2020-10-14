// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_HyperGraphBlockGroup_hpp
#define popsparse_HyperGraphBlockGroup_hpp

#include "HyperGraphBlock.hpp"
#include <unordered_map>

namespace popsparse {
namespace experimental {

/*
This is a base class for partitioning algorithms
that use memory estimation model for sparse matrix multiplication on IPU.
The model assumes that matmul and reduce verices are grouped by tiles based on
row and colum of matrix C and index of a partial they compute.

The following notation for matrix multiplication is used:
A[M,K] x B[K,N] = C[M,N]

The dimension K is split into P partials.
The matmul vertices are grouped alongside P,N,M dimensions (column-row split)
or P,M,N dimensions (row-column split)

Example:

C[4,4]:
P = 2
4 x 4 x 2 = 32 vertices

16 tiles
32 / 16 = 2 vertices per tile

column-row split:
partition 1:
 0  2  4  6
 0  2  4  6
 1  3  5  7
 1  3  5  7

partition 2:
 8 10 12 14
 8 10 12 14
 9 11 13 15
 9 11 13 15

row-column split:
partition 1:
 0  0  1  1
 2  2  3  3
 4  4  5  5
 6  7  7  7

partition 2:
 8  8  9  9
10 10 11 11
12 12 13 13
14 14 15 15

The reduce vertices are grouped alongside N,M dimensions (column-row split)
or M,N dimensions (row-column split)

The goal of the model is to estimate P
and also the case of either column-row or row-column split to minimize memory
usage on each tile for the maximum of two usages: matmul and reduce.

The model is based on a uniform sparsity distribution and even vertices
distribution. That is, for any case of sparse matrix split, we assume that the
number of non-zero blocks in every group is the same. Also, the number of
vertices on each tile is the same. It also assumes that we can split column and
row of every matrix as needed without leftovers.

Only a memory for the tensor that is needed for a vertex execution is computed.
That is, if some tensor needs to be copied to a different tile, only the copy
size is computed and the original tensor's contribution to the tile's memory
where it is located is out of scope here. It is handled by another logic also in
this class. The derived classes use the initial model estimation and implement
algorithms close to that scheme.

We have 4 cases for matmul:
1. Dense x Sparse = Dense, column-row split
2. Dense x Sparse = Dense, row-column split
3. Dense x Dense = Sparse, column-row split
4. Dense x Dense = Sparse, row-column split

And we have 2 cases for reduce:
1. Dense x Sparse = Dense
2. Dense x Dense = Sparse
We don't have two separate cases here for column-row or row-column split,
because we use partials groups on tiles already defined in matmul step
and we only try to place reduce vertices in most optimal way.

The following notations are used further in the description:

Wa  - memory taken by block on matrix A
Wb  - memory taken by block on matrix B
Wc  - memory taken by block on matrix C
Wv  - memory taken by partial result
Wv' - memory taken by partial result plus matmul vertex code
Wc' - memory taken by block on matrix C plus reduce vertex code
Y   - memory occupied on a tile

F   - number of non-zero elements
V   - total number of vertices
E   - number of vertices (matmul or reduce) on a tile

S - sparsity (share of non-zero blocks)
T - total number of tiles

1. Matlul case
1.1. Dense x Sparse = Dense, column-row split
=============================================

The memory occupied on any tile t on a column j and partition p is given by
formula: Yt(j,p) = {Sum by i in t}(Wv') + Wb * Fj,p + {Sum by i in t}(Wa * Fj,p)
(1)

where:
Fj,p - number of non-zero blocks in matrix B in column j and partition p
i    - vertex index

Because we group vertices by tiles in order: partition, column, row and we
assume even cuts, the tile only take vertices from the same partition and
column.

Because the blocks under the sums are the same for all rows i,  we can rewrite
(1) as:

Yt(j,p) = Wv' * Et + Wb * Fj,p + Wa * Fj,p * Et (2)

where:
Et - number of vertices on tile t

Because we assume that the number of vertices on every tile is the same
and number of non-zero blocks in matrix B is the same for every tile, we have:

V = M * N * P
Et = E = V / T = (M * N * P) / T (3)
(now and further we always assume that M * N * P >= T)
Fj,p = F = (K * S) / P   (4)

Combining (2),(3) and (4) and dropping indexes, we have:

Y = Wv' * M * N * P * / T + Wa * M * N * K * S / T + Wb * K * S / P (5)

Minimizing (5) w.r.t. P, we have:

Popt = Sqrt(Wb/Wv' * K * S * T/(M * N))

Because we consider only cases where number of non-zero blocks is >= 1,
we have an upper cinstraint: P <= K * S
and also apparent lowet constraint: P >= 1
Taking tose constraints into account, we have:

Popt = max(1, min(K * S, Sqrt(Wb/Wv' * K * S * T/(M * N)))) (6)

1.2. Dense x Sparse = Dense, row-column split
=============================================

The memory occupied on any tile t on a row i and partition p is given by
formula: Yt(i,p) = {Sum by j in t}(Wv') + {Sum by j in t}(Wb * Fj,p) + Wa * Ht,p
(7)

where:
Ht,p - number of rows in part p in matrix B where at least one block in columns
j on tile t is non-zero.

If we have a slice of matrix B, then the probability for each row of the slice
to have at least one non-zero block will be the probability of at least one
success of Bernoulli trial for the base probability of S and the number of tries
equal to the number of columns in the slice C and is given by formula:

beta(S, C) = 1 - pow((1 - S), C) (8)

Because here number of columns C is the same of the number pf vertices on a tile
E, for uniform sparsity case we have:

Ht,p = H = beta(S, E) * K / P (9)

Yt(i,p) = Y = Wv' * E + Wb * E * F + Wa * H (10)

and substituting from (3),(4) and (9):

Y = Wb * M * N * P / T + Wb * M * N * K * S / T + Wa * beta(S, M * N * P / T) *
K / P (11)

Finding the minimum of (11) is not possible analythically.
We find the minimum for two corner cases: beta() = S and beta() = 1

Popt1 = max(1, min(K * S, Sqrt(Wa/Wv' * K * S * T/(M * N)))) (12)
Popt2 = max(1, min(K * S, Sqrt(Wa/Wv' * K * T/(M * N))))     (13)

1.3. Dense x Dense = Sparse, column-row split
=============================================

The memory occupied on any tile t on a column j and partition p is given by
formula: Yt(j,p) = Wv' * Et + Wb * Lp + Wa * Lp * Et

where:
Lp - the length of part p

For sparse matrix C we have:
V = M * N * P * S

and therefore for the case of uniform sparsity:

Et = E = M * N * P * S / T (14)

also:
Lp = L = K / P (15)

Substituting (14),(15) and dropping inicies, we have:

Y = Wv' * M * N * S * P / T + Wa * M * N * K * S / T + Wb * K / P (16)

Minimizing (16) w.r.t. P, and considering the upper bound condition:
P <= K
we have:

Popt = max(1, min(K, Sqrt(Wb/Wv' * K * T/(M * N * S)))) (17)

1.4. Dense x Dense = Sparse, row-column split
=============================================

This case is very similar for the previous one.
The the formulas for the memory and optimal P:

Y = Wv' * M * N * S * P / T + Wb * M * N * K * S / T + Wa * K / P (18)
Popt = max(1, min(K, Sqrt(Wa/Wv' * K * T/(M * N * S))))           (19)

2. Reduce case
2.1. Dense x Sparse = Dense
=============================================

The memory occupied on any tile t for reduce step is given by formula:

Yt = Wc' * Et + Wv * P * Et (20)

where:
Et - number of reduce vertices on a tile t

The total number of reduce vertices is given by formula:

V = M * N

For even distribution of vertices by tiles we have:
Et = E = max(1, M * N / T) (21)

Substituting (21) into (20) and dropping tile index we have:

Y = (Wc' + Wv * P) * max(1, M * N / T) (22)

2.2. Dense x Dense = Sparse
=============================================

In this case the number of reduce vertices is different and is given by formula:

V = M * N * S

therefore we have:

E = max(1, M * N * S/ T) (23)

substituting (23) into (20) we have:

Y = (Wc' + Wv * P) * max(1, M * N * S / T) (24)

3. Optimal memory usage with respect to matmul and reduce
=========================================================

To estimate the memory consumption, we need to take the highest memory
estimation from matmul and reduce operations. That is, we should take the
maximum from the pairs of equations [(5),(22)], [(11),(22)] , [(16),(24)],
[(18),(24)]

Equalizing those pairs of equations will result in quadratic equations with
respect to P: a * x ^ 2 + b * x + c

The values for a, b, c are the following:
[(5),(22)]:

a = Wv' * M * N / T - Wv * max(1, M * N / T)              (25.1)
b = Wa * M * N * K * S / T - Wc' * max(1, M * N / T)      (25.2)
c = Wb * K * S                                            (25.3)

[(11),(22)]:

a = Wv' * M * N / T - Wv * max(1, M * N / T)              (26.1)
b = Wb * M * N * K * S / T - Wc' * max(1, M * N / T)      (26.2)
c = Wa * K * beta(S, M * N * P / T)                       (26.3)

Here we ignore beta dependency on P that makes b non constant
and we solve the equations for two corner cases as for (11)

[(16),(24)]:
a = Wv' * M * N * S / T - Wv * max(1, M * N * S / T)      (27.1)
b = Wa * M * N * K * S / T - Wc' * max(1, M * N * S / T)  (27.2)
c = Wb * K                                                (27.3)

[(18),(24)]:
a = Wv' * M * N * S / T - Wv * max(1, M * N * S / T)      (28.1)
b = Wa * M * N * K * S / T - Wc' * max(1, M * N * S / T)  (28.2)
c = Wa * K                                                (28.3)

Solving these equations give us the alternative points to check.
In practice usually matmul and reduce curves intersect only for cases when M, N
are small and K is big.

Finally, here is the simple formulas for the sparsity,
given by number of non-zero blocks:
===================================

Dense x Sparse = Dense:
S = Nz / (K * N)  (29)

Dense x Dense = Sparse:
S = Nz / (M * N)  (30)

The amount of computations can be reduced, if substituting S
in many formulas.
*/
class HyperGraphBlockGroup : public HyperGraphBlock {

public:
  HyperGraphBlockGroup(BlockMatrix &A, BlockMatrix &B,
                       poplar::Type inDataTypeIn, poplar::Type outDataTypeIn,
                       poplar::Type partialDataTypeIn, int nTileIn);

  virtual ~HyperGraphBlockGroup() = default;

  // Helper function
  // Given a list of n that we want to split to k parts as even as possible,
  // returns part number to which index i belongs
  static unsigned getPartNum(unsigned n, unsigned k, unsigned i);

protected:
  // Node weight in bytes
  struct W {
    // Temporary weight
    unsigned long long wTmp = 0ULL;
    // Total weight
    unsigned long long wTotal = 0ULL;
  };

  // Helper function
  // Fills nodes A,B to V mapping data structures
  void fillNodeAB2VMapping();

  // Helper function
  // Fills nodes C to V mapping data structures
  void fillNodeC2VMapping();

  // Tiles assignment layout
  // Used for logging/debugging
  struct TaDbg {
    TaDbg(unsigned numPartitionsIn, unsigned blockRowsC, unsigned blockColsC,
          unsigned blockColsA);

    void setA(unsigned blockRow, unsigned blockCol, int idTile);
    void setB(unsigned blockRow, unsigned blockCol, int idTile);
    void setC(unsigned blockRow, unsigned blockCol, int idTile);
    void setV(unsigned blockRow, unsigned blockCol, unsigned p, int idTile);

    const unsigned numPartitions;
    const unsigned numRowsC;
    const unsigned numColsC;
    const unsigned numColsA;

    std::vector<std::vector<std::vector<int>>> v;
    std::vector<std::vector<int>> a;
    std::vector<std::vector<int>> b;
    std::vector<std::vector<int>> c;
  };

  // Assigns nodes A, B to tiles
  void placeABNodes(const std::vector<std::vector<std::unordered_set<int>>>
                        &tilesByColAndPartition,
                    const std::vector<std::vector<std::unordered_set<int>>>
                        &tilesByRowAndPartition,
                    std::vector<W> &tilesWeights, TaDbg &taDbg);

  // Assigns nodes C to tiles
  void placeCNodes(std::vector<int> &tilesWeightsReduce, TaDbg &taDbg);

  void logTilesMapping(const TaDbg &taDbg);

  void getCalcParams(float &M, float &N, float &K, float &T, float &Nz,
                     float &Wv, float &Wa, float &Wb, float &Wc);

  // Initially estimates the number of k parts
  void estimateP();
  // Initially estimates the amount of memory on each tile
  void estimateTileW();

  bool isResultSparse;

  unsigned wA;
  unsigned wB;
  unsigned wV;
  unsigned wP;
  unsigned wC;

  enum class NodeType { A, B, C, V };

  std::vector<unsigned> nonZeroBlocksByCol;
  unsigned nonZeroBlocksTotal;

  // Mapping of nodes A to partitions
  std::unordered_map<unsigned, std::unordered_set<unsigned>> nodeAPartitions;
  // Mapping of nodes B to partitions
  std::unordered_map<unsigned, std::unordered_set<unsigned>> nodeBPartitions;
  // Mapping of nodes A, B to nodes V
  //________________<node V id,    <node A,B id, NodeType>>
  std::unordered_map<unsigned int, std::unordered_map<unsigned int, NodeType>>
      nodeVInputMapping;
  // Mapping of nodes V to nodes C
  //________________<node C id,               <node V id>>
  std::unordered_map<unsigned int, std::vector<unsigned int>> nodeCInputMapping;

  // number of partitions
  unsigned P;
  // true if doing row split, false if column split
  bool doRowSplit;

  // Initially estimated P for column split
  int estPColSplit;
  // Initially estimated P for row split (2 cases)
  std::array<int, 2> estPRowSplit;
  // P when matmul is equal to reduce for column split
  std::array<int, 2> estPColSplitMmEqRd = {-1, -1};
  // P when matmul is equal to reduce for row split
  std::array<int, 4> estPRowSplitMmEqRd = {-1, -1, -1, -1};
  // Initially estimated the anount of memory on each tile (in bytes) for column
  // split
  float wTileEstColSplit;
  // Initially estimated the anount of memory on each tile (in bytes) for row
  // split
  float wTileEstRowSplit;
};

} // namespace experimental
} // namespace popsparse

#endif