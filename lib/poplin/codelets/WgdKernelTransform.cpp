// Copyright (c) Graphcore Ltd, All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include <cassert>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace poplin {

template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
class WgdKernelTransform : public Vertex {

  /* Set this to true if transform is stored in transposed order */
  static constexpr bool transpose = true;

  /* storage depends on whether transpose or normal form of transform is
   * stored
   */
  FPType &wrTf(const unsigned base, const unsigned row, const unsigned col,
               const unsigned elem) {
    return transpose ? wTf[base + row * patchSizeY + col][elem]
                     : wTf[base + col * patchSizeX + row][elem];
  }

  FPType rdIn(unsigned base, unsigned row, unsigned col, unsigned elem) const {
    return wIn[base + col * kernelX + row][elem];
  }

public:
  /* Each input is a 1D vector of independent channels which may be a mix of
   * input and output channels. Therefore kernelCols*kernelRow vectors are
   * required to have all elements of a kernel. The 1D vectors are stored in row
   * order
   */
  Vector<Input<Vector<FPType>>, ONE_PTR> wIn;

  /* Same as wIn except that numOutCols*numOutRows vectors each of dimension
   * 1xdepth are stored
   */
  Vector<Output<Vector<FPType, ONE_PTR>>> wTf;

  bool compute() {
    const unsigned numOutCols = patchSizeY;
    const unsigned numOutRows = patchSizeX;
    const unsigned nGroups = wTf.size() / (numOutCols * numOutRows);
    assert(numOutCols == 4);
    assert(numOutRows == 4);
    assert(kernelX == 3);
    assert(kernelY == 3);

    for (int group = 0; group < nGroups; ++group) {
      unsigned gBaseIn = kernelY * kernelX * group;
      unsigned gBaseOut = numOutRows * numOutCols * group;

      const unsigned depth = wIn[0].size();

      for (unsigned elem = 0; elem < depth; ++elem) {
        FPType g[kernelX][kernelY];

        for (unsigned row = 0; row < kernelX; ++row) {
          for (unsigned col = 0; col < kernelY; ++col) {
            g[row][col] = rdIn(gBaseIn, row, col, elem);
          }
        }

        FPType A = (g[0][0] + g[0][1] + g[0][2]) * FPType(0.5);
        FPType B = (g[0][0] - g[0][1] + g[0][2]) * FPType(0.5);

        FPType C = (g[0][0] + g[1][0] + g[2][0]) * FPType(0.5);
        FPType F = (g[0][0] - g[1][0] + g[2][0]) * FPType(0.5);

        FPType D = (g[2][0] + g[2][1] + g[2][2]) * FPType(0.5);
        FPType E = (g[2][0] - g[2][1] + g[2][2]) * FPType(0.5);

        FPType G = (g[1][0] + g[1][1] + g[1][2]) * FPType(0.5);
        FPType H = (g[1][0] - g[1][1] + g[1][2]) * FPType(0.5);

        FPType I = (g[0][2] + g[1][2] + g[2][2]) * FPType(0.5);
        FPType J = (g[0][2] - g[1][2] + g[2][2]) * FPType(0.5);

        wrTf(gBaseOut, 0, 0, elem) = g[0][0];
        wrTf(gBaseOut, 0, 1, elem) = A;
        wrTf(gBaseOut, 0, 2, elem) = B;
        wrTf(gBaseOut, 0, 3, elem) = g[0][2];

        wrTf(gBaseOut, 1, 0, elem) = C;
        wrTf(gBaseOut, 1, 1, elem) = (A + G + D) * FPType(0.5);
        wrTf(gBaseOut, 1, 2, elem) = (B + H + E) * FPType(0.5);
        wrTf(gBaseOut, 1, 3, elem) = I;

        wrTf(gBaseOut, 2, 0, elem) = F;
        wrTf(gBaseOut, 2, 1, elem) = (A - G + D) * FPType(0.5);
        wrTf(gBaseOut, 2, 2, elem) = (B - H + E) * FPType(0.5);
        wrTf(gBaseOut, 2, 3, elem) = J;

        wrTf(gBaseOut, 3, 0, elem) = g[2][0];
        wrTf(gBaseOut, 3, 1, elem) = D;
        wrTf(gBaseOut, 3, 2, elem) = E;
        wrTf(gBaseOut, 3, 3, elem) = g[2][2];
      }
    }
    return true;
  }
};

template class WgdKernelTransform<float, 4, 4, 3, 3>;
template class WgdKernelTransform<half, 4, 4, 3, 3>;

} // namespace poplin
