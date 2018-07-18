#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

// Assembly implementation of popnn::NonLinearity2D vertex template instantiations.
#define NonLinearityHalfSigm2D __runCodelet_popnn__NonLinearity2D___half_0
#define NonLinearityFloatSigm2D __runCodelet_popnn__NonLinearity2D___float_0
#define NonLinearityHalfRelu2D __runCodelet_popnn__NonLinearity2D___half_1
#define NonLinearityFloatRelu2D __runCodelet_popnn__NonLinearity2D___float_1
#define NonLinearityHalfTanh2D __runCodelet_popnn__NonLinearity2D___half_2
#define NonLinearityFloatTanh2D __runCodelet_popnn__NonLinearity2D___float_2

// Restrictions
//
//  * Vertex state aligned to at least 4 bytes.

.text
/*****************************************************************************/
// popnn::NonLinearity2D<half, NON_LINEARITY_SIGMOID>
// popnn::NonLinearity2D<float, NON_LINEARITY_SIGMOID>
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
// popnn::NonLinearity2D<half, NON_LINEARITY_RELU>
// popnn::NonLinearity2D<float, NON_LINEARITY_RELU>
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
// popnn::NonLinearity2D<half, NON_LINEARITY_TANH>
// popnn::NonLinearity2D<float, NON_LINEARITY_TANH>
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
