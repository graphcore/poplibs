#ifndef popnn_NonLinearityInternal_hpp
#define popnn_NonLinearityInternal_hpp

// One hot / softmax scaling to improve accuracy
// Choosing scaling of (62000) means that accuracy is
// greatly improved compared to the default scaling of 1.0.
// The maximum we could use is 65504 which is the maximum number representatble
// in IEEE FP16 but taking it's exp(log(65504) + e) where e is a rounding error,
// could result in the number overflowing.
#define SOFTMAX_SCALING (62000.0F)

#endif // popnn_NonLinearityInternal_hpp
