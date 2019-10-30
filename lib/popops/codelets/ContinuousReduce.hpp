#include "ReduceCodelets.hpp"

namespace popops {

template <typename OutType, bool isUpdate>
using ROT =
    typename std::conditional<isUpdate, InOut<Vector<OutType, SCALED_PTR32, 4>>,
                              Output<Vector<OutType, SCALED_PTR32, 4>>>::type;

template <typename ReduceOp, typename PartialsType, typename OutType,
          bool isUpdate>
static constexpr bool useExternal() {
  bool externalOp = std::is_same<ReduceOp, ReduceAdd>::value ||
                    std::is_same<ReduceOp, ReduceSquareAdd>::value;
  bool externalTypes = (std::is_same<OutType, float>::value ||
                        std::is_same<OutType, half>::value) &&
                       (std::is_same<PartialsType, float>::value ||
                        std::is_same<PartialsType, half>::value);
  return externalOp && externalTypes;
}

} // namespace popops
