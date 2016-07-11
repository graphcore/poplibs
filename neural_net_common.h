#ifndef _neural_net_common_h_
#define _neural_net_common_h_

enum NonLinearityType {
  NON_LINEARITY_NONE,
  NON_LINEARITY_SIGMOID,
  NON_LINEARITY_RELU
};

enum LossType {
  SUM_SQUARED_LOSS,
  SOFTMAX_CROSS_ENTROPY_LOSS
};

enum ResidualMethod {
  RESIDUAL_NONE,
  RESIDUAL_PAD,
  RESIDUAL_WEIGHTED_CONV_IF_SIZES_DIFFER,
  RESIDUAL_WEIGHTED_CONV
};


#endif // _neural_net_common_h_
