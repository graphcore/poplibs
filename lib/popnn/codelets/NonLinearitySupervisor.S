#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

// Assembly implementation of popnn::NonLinearitySupervisor vertex template variations.
#define NonLinearityHalfSigmSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___half_popnn__NonLinearityType__SIGMOID
#define NonLinearityFloatSigmSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___float_popnn__NonLinearityType__SIGMOID
#define NonLinearityHalfReluSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___half_popnn__NonLinearityType__RELU
#define NonLinearityFloatReluSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___float_popnn__NonLinearityType__RELU
#define NonLinearityHalfTanhSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___half_popnn__NonLinearityType__TANH
#define NonLinearityFloatTanhSupervisor \
  __runCodelet_popnn__NonLinearitySupervisor___float_popnn__NonLinearityType__TANH

// Restrictions
//
//  * At least 32-bit aligned source/destination address.
//
//  Currently no memory constraints that would allow 128-bit loads for wider
//  vector operations.

/*****************************************************************************/
// popnn::NonLinearitySupervisor<half, popnn::NonLinearityType::SIGMOID>
// popnn::NonLinearitySupervisor<float, popnn::NonLinearityType::SIGMOID>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfSigmSupervisor
#define FLOAT_SYMBOL_NAME NonLinearityFloatSigmSupervisor
#define OP_UNIQUE_PREFIX sigm
#define CALC_HALF(dst, src) f16v2sigm dst, src
#define CALC_FLOAT(dst, src) f32sigm dst, src
#include "detail/NonLinearitySupervisorTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

/*****************************************************************************/
// popnn::NonLinearitySupervisor<half, popnn::NonLinearityType::RELU>
// popnn::NonLinearitySupervisor<float, popnn::NonLinearityType::RELU>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfReluSupervisor
#define FLOAT_SYMBOL_NAME NonLinearityFloatReluSupervisor
#define OP_UNIQUE_PREFIX relu
#define CALC_HALF(dst, src) f16v2max dst, src, $azero
#define CALC_FLOAT(dst, src) f32max dst, src, $azero
#include "detail/NonLinearitySupervisorTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

/*****************************************************************************/
// popnn::NonLinearitySupervisor<half, popnn::NonLinearityType::TANH>
// popnn::NonLinearitySupervisor<float, popnn::NonLinearityType::TANH>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfTanhSupervisor
#define FLOAT_SYMBOL_NAME NonLinearityFloatTanhSupervisor
#define OP_UNIQUE_PREFIX tanh
#define CALC_HALF(dst, src) f16v2tanh dst, src
#define CALC_FLOAT(dst, src) f32tanh dst, src
#include "detail/NonLinearitySupervisorTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

#endif // __IPU__
