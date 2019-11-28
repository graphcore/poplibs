#include "RandomUtils.hpp"

namespace poprand {

class SetSeedSupervisor : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  SetSeedSupervisor();

  Input<Vector<unsigned, ONE_PTR, 8>> seed;
  const uint32_t seedModifierUser;
  const uint32_t seedModifierHw;

  IS_EXTERNAL_CODELET(true);

  bool compute() { return true; }
};

} // namespace poprand
