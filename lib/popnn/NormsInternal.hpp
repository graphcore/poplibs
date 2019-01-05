// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_NormsInternal_hpp
#define popnn_NormsInternal_hpp

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>


namespace popnn {

// Once T6054 is fixed, the common functions used by normalisations currently
// in poplib/Norms.hpp must be moved here. The description below details the
// need for rearrangement of tensors used by different normalisation functions
//
// The dimension of the batch statistics and parameters in normalisations
// depend on the type of normalisation. Activations/gradients have shape
// [B][C][..F..] on the external interface but have to be rearranged internally
// when operations to be done on them are related to statistics computation
// (eg: whitening and gradient flow through the statistics).
//  Where B = batch size
//        C = number of channels
//        ..F.. = field dimensions
//
// Functions related to statistics : whitening, gradient flow through statistics
// [B][C][..F..] is transformed to [N1][C1][..F..]
//   where N1 and C1 depend on the normalisation type
// For Batch Norm :    N1 = B, C1 = C
// For Group Norm :    N1 = G, C1 = B * C / G
// For Layer Norm :    N1 = C, C1 = B
// For Instance Norm : N1 = 1, C1 = B * C
//
// Operations using parameters (beta and gamma) use tensor with shape
// [B][C][..F..]


// consistency check on input tensors used in normalisations. The shape of
// of the input activations is [B][C][..F..] where
//    B is the batch size
//    C is the number of channels
//    ..F.. are the field dims. ..F.. is absent for fully connected layers
void checkTensorShape(poplar::Tensor in);

// The shapes the input tensor to have shape required by the normalisation
// functions. Can be removed once T6054 is fixed.
poplar::Tensor preProcessNormActs(const poplar::Tensor &acts);

// This shapes the acts tensor to the shape required by the external functions
poplar::Tensor postProcessNormActs(const poplar::Tensor &acts,
                                   unsigned originalActsRank);

} // namespace popnn


#endif // #ifndef popnn_NormsInternal_hpp
