#ifndef _VertexOptim_hpp_
#define _VertexOptim_hpp_

namespace popconv {

// Whether or not to use delta pointers to convolution partials depending on
// the number of edges. Delta pointers use less memory but are slightly slower.
// By default they are always used.
static inline bool useDeltaEdgesForConvPartials(unsigned numEdges) {
  (void)numEdges;
  return true;
}

}

#endif
