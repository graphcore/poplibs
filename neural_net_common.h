#ifndef _neural_net_common_h_
#define _neural_net_common_h_

typedef enum NonLinearityType {
  NON_LINEARITY_NONE,
  NON_LINEARITY_SIGMOID,
  NON_LINEARITY_RELU
} NonLinearityType;

typedef enum LossType {
  SUM_SQUARED_LOSS,
  SOFTMAX_CROSS_ENTROPY_LOSS
} LossType;

typedef enum NormalizationType {
  NORMALIZATION_NONE,
  NORMALIZATION_LR
} NormalizationType;

#define FPType float
#define _STR0(X) #X
#define _STR(X) _STR0(X)
#define FPTypeStr _STR(FPType)

#endif // _neural_net_common_h_
