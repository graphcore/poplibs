// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_GeneralMatrixAdd_hpp
#define poplibs_test_GeneralMatrixAdd_hpp

#include <boost/multi_array.hpp>

namespace poplibs_test {
namespace axpby {

/*
 * Computes matD = alpha * op(matA) + beta * op(matB)
 *
 * where op(matA) = A     if transposeA = false
 *       op(matA) = A'    if transposeA = true
 *
 *
 *       op(matB) = B     if transposeB = false
 *       op(matB) = B'    if transposeB = true
 *
 * Matrix dimensions of op(A) must be equal to op(B)
 */

void add(const boost::multi_array_ref<double, 2> matA,
         const boost::multi_array_ref<double, 2> matB,
         boost::multi_array_ref<double, 2> matC, float alpha = 1.0,
         float beta = 1.0, bool transposeA = false, bool transposeB = false);

void add(const boost::multi_array_ref<double, 1> matA,
         const boost::multi_array_ref<double, 1> matB,
         boost::multi_array_ref<double, 1> matC, float alpha = 1.0,
         float beta = 1.0);

} // End namespace axpby.
} // End namespace poplibs_test.

#endif // poplibs_test_GeneralMatrixAdd_hpp
