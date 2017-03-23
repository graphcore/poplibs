#ifndef popnn_ref_gemm_hpp_
#define popnn_ref_gemm_hpp_

#include<boost/multi_array.hpp>

namespace ref {
namespace gemm {

/*
 * Computes matD = beta * matC + alpha * op(matA) * op(matB)
 *
 * where op(matA) = A     if transposeA = false
 *       op(matA) = A'    if transposeA = true
 *
 *
 *       op(matB) = B     if transposeB = false
 *       op(matB) = B'    if transposeB = true
 *
 */

void generalMatrixMultiply(
            const boost::multi_array<double, 2> &matA,
            const boost::multi_array<double, 2> &matB,
            const boost::multi_array<double, 2> &matC,
            boost::multi_array<double, 2> &matD,
            float alpha,
            float beta,
            bool  transposeA,
            bool  transposeB);

} // End namespace gemm.
} // End namespace ref.

#endif  // popnn_ref_FullyConnected_hpp_
