#ifndef popnn_NonLinearityInternal_hpp
#define popnn_NonLinearityInternal_hpp


// One hot / softmax scaling to improve accuracy
// Choosing scaling of max half (65504) means that accuracy is
// greatly improved compared to the default scaling of 1.0.
#define SOFTMAX_SCALING (65504.0F)

#endif // popnn_NonLinearityInternal_hpp
