#include <popops/codelets.hpp>
#include "popopsCycleEstimators.hpp"
#include "reduction/CycleEstimationFunctions.hpp"
#include "reduction/ReductionVertex.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include <fstream>

namespace popops {

static std::string findGraphProg() {
  // TODO: This needs to be replaced with a proper object search mechanism
  // in poplar.
  const auto env = std::getenv("IPU_POPOPS_GP");
  if (env && std::ifstream(env).good())
    return env;

#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + "popops.gp";
    return path;
  }
#endif

  std::string path = "lib/popops/popops.gp";
  if (std::ifstream(path).good())
    return path;

  path = "../" + path;
  return path;
}

void addReduceCodelets(poplar::Graph &graph) {

  typedef std::vector<std::pair<poplar::Type, poplar::Type>> type_pairs;

  type_pairs fullTypes = {
    {poplar::FLOAT, poplar::FLOAT},
    {poplar::HALF, poplar::FLOAT},
    {poplar::FLOAT, poplar::HALF},
    {poplar::HALF, poplar::HALF},
    {poplar::INT, poplar::INT},
  };

  type_pairs equalTypes = {
    {poplar::FLOAT, poplar::FLOAT},
    {poplar::HALF, poplar::HALF},
    {poplar::INT, poplar::INT},
  };

  type_pairs boolTypes = {
    {poplar::BOOL, poplar::BOOL},
  };

  auto registerReduceCycleEstimators = [&](
                                       const type_pairs& types,
                                       popops::Operation operation
                                       ) {
    using std::placeholders::_1;
    using std::placeholders::_2;

    for (const auto &p : types) {
      for (bool isScale : {false, true}) {
        for (bool isUpdate : {false, true}) {
          for (bool partialsAreOutputSize : {false, true}) {
            std::string opName = getReductionVertexOpName(operation);
            auto vertexName = getReductionVertexName(opName, p.first, p.second,
                                                     isScale, isUpdate,
                                                     partialsAreOutputSize);
            graph.registerCycleEstimator(
                  vertexName,
                  std::bind(getCycleEstimateForReduceVertex, _1, _2,
                            p.first, p.second, operation, isUpdate, isScale)
            );
          }
        }
      }
    }
  };

  registerReduceCycleEstimators(fullTypes, popops::Operation::ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::SQUARE_ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::MUL);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MAX);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MIN);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_AND);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_OR);
}

void addCodelets(poplar::Graph &graph) {
  graph.addCodelets(findGraphProg());
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
  addReduceCodelets(graph);
}

} // namespace popreduce
