// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifdef __IPU__
#include <ipu_builtins.h>
#endif

#include <poplar/Vertex.hpp>

using namespace poplar;

namespace poplin {
namespace experimental {

template <unsigned align> class LUDCoreVertex : public MultiVertex {
public:
  LUDCoreVertex();
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLU;

  int depth;

  bool compute(unsigned worker_id) {

#ifdef __IPU__
    if (align == 8) {
      for (int i = 0; i + 1 < depth; i += 2) {
        for (int y = i + 1 + worker_id; y < depth;
             y += MultiVertex::numWorkers()) {
          auto *rhs = (float2 *)&sliceLU[i][i];
          auto *result = (float2 *)&sliceLU[y][i];
          const int loops = ((depth - (i + 1)) / 2);
          asm volatile(
              R"(
                ld64 $a4:5, $mzero, %[result], 0 # Load lhs and first target val.
                ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load i,i and first rhs
                f32div $a4, $a4, $a2
                f32mul $a1, $a4, $a3
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load first rhs
                  f32sub $a5, $a5, $a1
                }
                {
                  st64step $a4:5, $mzero, %[result]+=, 1 # Store first result
                  f32add $a6, $a4, $azero # Move lhs to it's target register
                }
                {
                  rpt %[loops], (2f - 1f) / 8 - 1
                  fnop
                }
                1:
                {
                  ld64step $a4:5, $mzero, %[result]+=, 0 # Load target val.
                  f32v2mul $a0:1, $a6:B, $a2:3
                }
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load next rhs
                  f32v2sub $a0:1, $a4:5, $a0:1
                }
                {
                  st64step $a0:1, $mzero, %[result]+=, 1 # Store last result
                  fnop
                }
                2:
            )"
              : [result] "+r"(result)
              : [loops] "r"(loops), [rhs] "r"(rhs)
              : "$a0:1", "$a2:3", "$a4:5", "$a6", "memory");
        }
        int j = i + 1;
        for (int y = j + 1 + worker_id; y < depth;
             y += MultiVertex::numWorkers()) {
          auto *rhs = (float2 *)&sliceLU[j][j];
          auto *result = (float2 *)&sliceLU[y][j];
          const int loops = ((depth - (j + 1)) / 2);
          asm volatile(
              R"(
                ld32 $a4, $mzero, %[result], 0 # Load lhs
                ld32step $a2, $mzero, %[rhs]+=, 1 # Load i,i
                f32div $a4, $a4, $a2
                {
                  st32step $a4, $mzero, %[result]+=, 1 # Store first result
                  f32add $a6, $a4, $azero # Move lhs to it's target register
                }
                ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load first rhs
                {
                  rpt %[loops], (2f - 1f) / 8 - 1
                  fnop
                }
                1:
                {
                  ld64step $a4:5, $mzero, %[result]+=, 0 # Load target val.
                  f32v2mul $a0:1, $a6:B, $a2:3
                }
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load next rhs
                  f32v2sub $a0:1, $a4:5, $a0:1
                }
                {
                  st64step $a0:1, $mzero, %[result]+=, 1 # Store last result
                  fnop
                }
                2:
            )"
              : [result] "+r"(result)
              : [loops] "r"(loops), [rhs] "r"(rhs)
              : "$a0:1", "$a2:3", "$a4:5", "$a6", "memory");
        }
      }
    } else {
      for (int i = 0; i < depth; i++) {
        for (int y = worker_id + 1 + i; y < depth;
             y += MultiVertex::numWorkers()) {
          sliceLU[y][i] /= sliceLU[i][i];
          for (int x = i + 1; x < depth; x++) {
            sliceLU[y][x] -= sliceLU[y][i] * sliceLU[i][x];
          }
        }
      }
    }
#else

    if (worker_id == 0) {
      for (int i = 0; i < depth; i++) {
        for (int y = 1 + i; y < depth; y++) {
          sliceLU[y][i] /= sliceLU[i][i];
          for (int x = i + 1; x < depth; x++) {
            sliceLU[y][x] -= sliceLU[y][i] * sliceLU[i][x];
          }
        }
      }
    }

#endif

    return true;
  }
};

template <unsigned align> class LUDRowVertex : public MultiVertex {
public:
  LUDRowVertex();
  Input<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLUCore;
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLU;

  int depth;
  int width;

  bool compute(unsigned worker_id) {
#ifdef __IPU__
    if (align == 8) {
      for (int i = 0; i < depth; i++) {
        const int loops = (width / 2);
        for (int y = i + 1 + worker_id; y < depth;
             y += MultiVertex::numWorkers()) {
          float lhs = sliceLUCore[y][i];
          auto *rhs = (float2 *)&sliceLU[i][0];
          auto *result = (float2 *)&sliceLU[y][0];
          asm volatile(
              R"(
                ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load first rhs
                {
                  rpt %[loops], (2f - 1f) / 8 - 1
                  fnop
                }
                1:
                {
                  ld64step $a4:5, $mzero, %[result]+=, 0 # Load target val.
                  f32v2mul $a0:1, %[lhs]:B, $a2:3
                }
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load next rhs
                  f32v2sub $a0:1, $a4:5, $a0:1
                }
                {
                  st64step $a0:1, $mzero, %[result]+=, 1 # Store last result
                  fnop
                }
                2:
            )"
              : [result] "+r"(result)
              : [loops] "r"(loops), [rhs] "r"(rhs), [lhs] "r"(lhs)
              : "$a0:1", "$a2:3", "$a4:5", "memory");
        }
      }
    } else {
      for (int i = 0; i < depth; i++) {
        for (int y = i + worker_id + 1; y < depth;
             y += MultiVertex::numWorkers()) {
          for (int x = 0; x < width; x++) {
            sliceLU[y][x] -= sliceLUCore[y][i] * sliceLU[i][x];
          }
        }
      }
    }

#else

    if (worker_id == 0) {
      for (int i = 0; i < depth; i++) {
        for (int y = 1 + i; y < depth; y++) {
          for (int x = 0; x < width; x++) {
            sliceLU[y][x] -= sliceLUCore[y][i] * sliceLU[i][x];
          }
        }
      }
    }

#endif
    return true;
  }
};

template <unsigned align> class LUDColVertex : public MultiVertex {
public:
  LUDColVertex();
  Input<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLUCore;
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLU;

  int depth;
  int height;

  bool compute(unsigned worker_id) {
#ifdef __IPU__
    if (align == 8) {
      for (int i = 0; i + 1 < depth; i += 2) {
        for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
          auto *rhs = (float2 *)&sliceLUCore[i][i];
          auto *result = (float2 *)&sliceLU[y][i];
          const int loops = ((depth - (i + 1)) / 2);
          asm volatile(
              R"(
                ld64 $a4:5, $mzero, %[result], 0 # Load lhs and first target val.
                ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load i,i and first rhs
                f32div $a4, $a4, $a2
                f32mul $a1, $a4, $a3
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load first rhs
                  f32sub $a5, $a5, $a1
                }
                {
                  st64step $a4:5, $mzero, %[result]+=, 1 # Store first result
                  f32add $a6, $a4, $azero # Move lhs to it's target register
                }
                {
                  rpt %[loops], (2f - 1f) / 8 - 1
                  fnop
                }
                1:
                {
                  ld64step $a4:5, $mzero, %[result]+=, 0 # Load target val.
                  f32v2mul $a0:1, $a6:B, $a2:3
                }
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load next rhs
                  f32v2sub $a0:1, $a4:5, $a0:1
                }
                {
                  st64step $a0:1, $mzero, %[result]+=, 1 # Store last result
                  fnop
                }
                2:
            )"
              : [result] "+r"(result)
              : [loops] "r"(loops), [rhs] "r"(rhs)
              : "$a0:1", "$a2:3", "$a4:5", "$a6", "memory");
        }
        int j = i + 1;
        for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
          auto *rhs = (float2 *)&sliceLUCore[j][j];
          auto *result = (float2 *)&sliceLU[y][j];
          const int loops = ((depth - (j + 1)) / 2);
          asm volatile(
              R"(
                ld32 $a4, $mzero, %[result], 0 # Load lhs
                ld32step $a2, $mzero, %[rhs]+=, 1 # Load i,i
                f32div $a4, $a4, $a2
                {
                  st32step $a4, $mzero, %[result]+=, 1 # Store first result
                  f32add $a6, $a4, $azero # Move lhs to it's target register
                }
                ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load first rhs
                {
                  rpt %[loops], (2f - 1f) / 8 - 1
                  fnop
                }
                1:
                {
                  ld64step $a4:5, $mzero, %[result]+=, 0 # Load target val.
                  f32v2mul $a0:1, $a6:B, $a2:3
                }
                {
                  ld64step $a2:3, $mzero, %[rhs]+=, 1 # Load next rhs
                  f32v2sub $a0:1, $a4:5, $a0:1
                }
                {
                  st64step $a0:1, $mzero, %[result]+=, 1 # Store last result
                  fnop
                }
                2:
            )"
              : [result] "+r"(result)
              : [loops] "r"(loops), [rhs] "r"(rhs)
              : "$a0:1", "$a2:3", "$a4:5", "$a6", "memory");
        }
      }
    } else {
      for (int i = 0; i < depth; i++) {
        for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
          sliceLU[y][i] /= sliceLUCore[i][i];
          for (int x = i + 1; x < depth; x++) {
            sliceLU[y][x] -= sliceLU[y][i] * sliceLUCore[i][x];
          }
        }
      }
    }

#else

    if (worker_id == 0) {
      for (int i = 0; i < depth; i++) {
        for (int y = 0; y < height; y++) {
          sliceLU[y][i] /= sliceLUCore[i][i];
          for (int x = i + 1; x < depth; x++) {
            sliceLU[y][x] -= sliceLU[y][i] * sliceLUCore[i][x];
          }
        }
      }
    }

#endif

    return true;
  }
};

// TODO: Add f32sisoslic and f32sisoamp implementations
template <unsigned align>
class [[poplar::constraint(
    "elem(**sliceLUCol) != elem(**sliceLURow)")]] LUDBlockVertex
    : public MultiVertex {
public:
  LUDBlockVertex();

  Input<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLURow;
  Input<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLUCol;
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS, align>>
      sliceLU;

  int depth;
  int width;
  int height;

  bool compute(unsigned worker_id) {

#ifdef __IPU__
    if (align == 8) {
      static constexpr unsigned aaccMask = CSR_W_FP_CLR__ZAACC__MASK
                                           << CSR_W_FP_CLR__ZAACC__SHIFT;
      const int loops = depth / 2 - 1;
      for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
        auto *lhs = (float2 *)&sliceLUCol[y][0];
        auto *result = &sliceLU[y][0];
        for (int x = 0; x < width; x++) {
          auto *rhs = (float2 *)&sliceLURow[x][0];
          auto packed_ptr = __builtin_ipu_tapack(rhs, lhs, 0);
          asm volatile(
              R"(
              {
                ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                setzi $a6, %[aaccMask]
              }
              #Loop begin
              .align 8
              {
                rpt %[loops], 0
                fnop
              }
              {
                ld2x64pace $a2:3, $a4:5, %[ptr]+=, $mzero, 0
                f32v2mac $a2:3, $a4:5
              }
              #Loop end
              {
                ld32 $a4, %[result], 0
                f32v2mac $a2:3, $a4:5
              }
              f32v2gina $a2:3, $azeros, 0
              f32sub $a4, $a4, $a2
              f32sub $a4, $a4, $a3
              st32step $a4, $mzero, %[result]+=, 1 # Store result
              )"
              : [ptr] "+r"(packed_ptr), [result] "+r"(result)
              : [loops] "r"(loops), [aaccMask] "n"(aaccMask)
              : "$a2:3", "$a4:5", "$a6", "memory");
        }
      }
    } else {
      for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
        for (int x = 0; x < width; x++) {
          for (int i = 0; i < depth; i++) {
            sliceLU[y][x] -= sliceLUCol[y][i] * sliceLURow[x][i];
          }
        }
      }
    }

#else

    for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
      for (int x = 0; x < width; x++) {
        for (int i = 0; i < depth; i++) {
          sliceLU[y][x] -= sliceLUCol[y][i] * sliceLURow[x][i];
        }
      }
    }

#endif

    return true;
  }
};

class LUDCoreSplitVertex : public MultiVertex {
public:
  LUDCoreSplitVertex();
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS>> sliceLU;
  Output<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS>> sliceL;

  int width;
  int height;

  bool compute(unsigned worker_id) {
    for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
      int end = width < y ? width : y;
      for (int x = 0; x < end; x++) {
        sliceL[y][x] = sliceLU[y][x];
        sliceLU[y][x] = 0;
      }
      if (y < width) {
        sliceL[y][y] = 1.0f;
      }
      for (int x = y + 1; x < width; x++) {
        sliceL[y][x] = 0;
      }
    }
    return true;
  }
};

class LUDBlockSplitVertex : public MultiVertex {
public:
  LUDBlockSplitVertex();
  InOut<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS>> sliceLU;
  Output<VectorList<float, poplar::VectorListLayout::DELTANELEMENTS>> sliceL;

  int width;
  int height;

  bool compute(unsigned worker_id) {
    for (int y = worker_id; y < height; y += MultiVertex::numWorkers()) {
      for (int x = 0; x < width; x++) {
        sliceL[y][x] = sliceLU[y][x];
        sliceLU[y][x] = 0;
      }
    }
    return true;
  }
};

template class LUDRowVertex<4>;
template class LUDRowVertex<8>;
template class LUDColVertex<4>;
template class LUDColVertex<8>;
template class LUDCoreVertex<4>;
template class LUDCoreVertex<8>;
template class LUDBlockVertex<4>;
template class LUDBlockVertex<8>;

} // namespace experimental
} // namespace poplin
