#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

// Assembly implementation of popnn::NonLinearity2D vertex template instantiations.
#define NonLinearityHalfSigm2D \
  __runCodelet_popnn__NonLinearity2D___half_popnn__NonLinearityType__SIGMOID
#define NonLinearityFloatSigm2D \
  __runCodelet_popnn__NonLinearity2D___float_popnn__NonLinearityType__SIGMOID
#define NonLinearityHalfRelu2D \
  __runCodelet_popnn__NonLinearity2D___half_popnn__NonLinearityType__RELU
#define NonLinearityFloatRelu2D \
  __runCodelet_popnn__NonLinearity2D___float_popnn__NonLinearityType__RELU
#define NonLinearityHalfTanh2D \
  __runCodelet_popnn__NonLinearity2D___half_popnn__NonLinearityType__TANH
#define NonLinearityFloatTanh2D \
  __runCodelet_popnn__NonLinearity2D___float_popnn__NonLinearityType__TANH

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.

/*****************************************************************************/
// popnn::NonLinearity2D<half, popnn::NonLinearityType::SIGMOID>
// popnn::NonLinearity2D<float, popnn::NonLinearityType::SIGMOID>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfSigm2D
#define FLOAT_SYMBOL_NAME NonLinearityFloatSigm2D
#define OP_UNIQUE_PREFIX sigm
#define CALC_HALF(dst, src) f16v2sigm dst, src
#define CALC_FLOAT(dst, src) f32sigm dst, src
#include "detail/NonLinearity2DTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

/*****************************************************************************/
// popnn::NonLinearity2D<half, popnn::NonLinearityType::RELU>
// popnn::NonLinearity2D<float, popnn::NonLinearityType::RELU>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfRelu2D
#define FLOAT_SYMBOL_NAME NonLinearityFloatRelu2D
#define OP_UNIQUE_PREFIX relu
#define CALC_HALF(dst, src) f16v2max dst, src, $azero
#define CALC_FLOAT(dst, src) f32max dst, src, $azero
#include "detail/NonLinearity2DTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

/*****************************************************************************/
// popnn::NonLinearity2D<half, popnn::NonLinearityType::TANH>
// popnn::NonLinearity2D<float, popnn::NonLinearityType::TANH>
/*****************************************************************************/
#define HALF_SYMBOL_NAME NonLinearityHalfTanh2D
#define FLOAT_SYMBOL_NAME NonLinearityFloatTanh2D
#define OP_UNIQUE_PREFIX tanh
#define CALC_HALF(dst, src) f16v2tanh dst, src
#define CALC_FLOAT(dst, src) f32tanh dst, src
#include "detail/NonLinearity2DTemplates.S"
#undef HALF_SYMBOL_NAME
#undef FLOAT_SYMBOL_NAME
#undef OP_UNIQUE_PREFIX
#undef CALC_HALF
#undef CALC_FLOAT

#endif // __IPU__
