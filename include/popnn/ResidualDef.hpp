#ifndef __residualdef_hpp__
#define __residualdef_hpp__

enum ResidualMethod {
  RESIDUAL_PAD,  // pad channel dimension, stride x/y dimentions
  RESIDUAL_CONCATENATE //concatenate channel dimensions, x/y must match
};

#endif // __residualdef_hpp__
