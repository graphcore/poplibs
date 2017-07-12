#ifndef _VertexOptim_hpp_
#define _VertexOptim_hpp_

static inline bool useDeltaEdgesForConvPartials(unsigned numEdges) {
  return numEdges >= 0;
}
#endif
