#include "ChannelOps.hpp"
#include "popops/PopopsChannelOps.hpp"

#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include "ConvUtilInternal.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace poplin {

void addToChannel(Graph &graph,
                  const Tensor &actsUngrouped,
                  const Tensor &addend,
                  float scale,
                  boost::variant<ComputeSet&, Sequence &> csOrProg,
                  const std::string debugPrefix)  {
  const auto fnPrefix = debugPrefix + "/addToChannel";
  const bool isProg = csOrProg.which() == 1;
  auto cs = isProg ? graph.addComputeSet(fnPrefix) :
                     boost::get<ComputeSet&>(csOrProg);

  // Convert actsUngrouped back into its internal layout, which matches
  // the in-memory layout. It is [G][C1][N]...[C2] where C2 is a nice
  // number like 8 or 16. N is the batch dimension, ... are the spatial
  // dimensions, G is the conv group dimension and C1 is the remaining channel
  // dimensions. Also, the [G] dimension is removed because it is 1, so the
  // shape is now [C1][N]...[C2]

  const auto acts =
      splitActivationChanGroups(graph,
        actsToInternalShape(actsUngrouped, 1, actsUngrouped.dim(1))
      )[0];

  // The number of channels in the inner-most dimension (i.e. adjacent in
  // memory). This is C2.
  const auto outChansPerGroup = acts.dim(acts.rank() - 1);
  // Reshape addend so that addendByGroup[i] is the i'th C2-sized group. The
  // shape should be [C1][C2].
  const auto addendByGroup =
      addend.reshape({addend.numElements() / outChansPerGroup,
                      outChansPerGroup});

  assert(addendByGroup.rank() == 2);
  assert(addendByGroup.dim(0) == acts.dim(0));
  assert(addendByGroup.dim(1) == outChansPerGroup);

  broadcastAddVectorInnermostInPlace(graph, acts, addendByGroup, scale, cs);
  if (isProg) {
    auto &prog = boost::get<Sequence &>(csOrProg);
    prog.add(Execute(cs));
  }
}

Tensor channelMul(Graph &graph,
                  const Tensor &actsUngrouped,
                  const Tensor &scale,
                  boost::variant<ComputeSet&, Sequence&> csOrProg,
                  const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/channelMul";
  const bool isProg = csOrProg.which() == 1;
  auto cs = isProg ? graph.addComputeSet(fnPrefix) :
                     boost::get<ComputeSet&>(csOrProg);

  auto actsOutUngrouped = graph.clone(actsUngrouped, fnPrefix + "/actsIn");
  const auto acts =
      splitActivationChanGroups(graph,
        actsToInternalShape(actsUngrouped, 1, actsUngrouped.dim(1))
      )[0];
  const auto actsOut =
      splitActivationChanGroups(graph,
        actsToInternalShape(actsOutUngrouped, 1, actsOutUngrouped.dim(1))
      )[0];
  const auto outChansPerGroup = acts.dim(acts.rank() - 1);

  const auto scaleByGroup =
      scale.reshape({scale.numElements() / outChansPerGroup,
                      outChansPerGroup});
  broadcastMulVectorInnermost(graph, acts, actsOut, scaleByGroup, cs, fnPrefix);
  if (isProg) {
    auto &prog = boost::get<Sequence &>(csOrProg);
    prog.add(Execute(cs));
  }
  return actsOutUngrouped;
}

} // namespace poplin
