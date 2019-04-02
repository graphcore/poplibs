#ifndef _performance_estimation_h_
#define _performance_estimation_h_

using namespace poplar;

enum class BinaryOpMethod {
  COPY_BROADCAST,
  VECTOR_BROADCAST,
  CHANNEL_OP,
  BROADCAST_AND_CHANNEL_OP
};

struct VertexInfo {
  unsigned vertices;
  unsigned slices;
  unsigned addendLen;
};

struct Costs{
  BinaryOpMethod method;
  std::uint64_t copy;
  std::uint64_t vertices;
};

// Simple cost estimates to compare methods of implementing basic binary
// operations.  The principle is that by counting the total number of vertices,
// tensor slices and the whole data size we can make a comparitive measure
// of cycles required for each method. Figures for the overhead in
// initial execution of a vertex, the loop per slice it processes and then
// the efficiency which data is processed once in the inner loop are taken
// from the codelet cycle estimators.

static Costs simpleBinaryOpCostEstimate(BinaryOpMethod method,
                                const VertexInfo info,
                                const std::vector<unsigned> &dimsShuffled,
                                unsigned matchingDim,
                                Tensor in1,
                                const Target &target)  {
  std::uint64_t copy = 0, vertices;
  const auto dimsIn1 = in1.shape();
  const unsigned dataSize = std::accumulate(dimsIn1.begin(), dimsIn1.end(), 1,
                                                std::multiplies<unsigned>());
  const unsigned dataSize2 = dataSize / dimsIn1[matchingDim];
  const auto vectorWidth = target.getVectorWidth(in1.elementType());

  switch (method) {
    case BinaryOpMethod::BROADCAST_AND_CHANNEL_OP:
      copy =  dataSize2/vectorWidth + 20 * dimsIn1[dimsShuffled[in1.rank()-1]];
      // Fall through
    case BinaryOpMethod::CHANNEL_OP:
      vertices = info.vertices * 39 +
                 dataSize/vectorWidth +
                 info.slices * 10;
      if(info.addendLen>2048)
        vertices += dataSize * 8;
      break;

    case BinaryOpMethod::VECTOR_BROADCAST:
      copy = 0;
      vertices = info.vertices * 20 +
                 6 * ((dataSize + vectorWidth - 1)/vectorWidth) +
                 info.slices * 28;
       break;

    case BinaryOpMethod::COPY_BROADCAST:
      vertices = info.vertices * 20 +
                 (dataSize * 6)/vectorWidth +
                 info.slices * 28;
      unsigned copySlices = 0;

      for(unsigned i = 0; i < dimsShuffled.size(); i++)
      {
          if(matchingDim == dimsShuffled[i])
            copySlices = 1;
          else
            copySlices *= in1.dim(dimsShuffled[i]);
      }
      copy  = matchingDim == dimsShuffled.back() ?
                          4 * dataSize/vectorWidth : 24 * dataSize/vectorWidth;
      copy += dataSize2 * 10;
      break;
  }
  return {method, copy, vertices};
}
#endif
