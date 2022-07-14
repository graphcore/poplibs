// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef poplibs_ExpressionGenerator_hpp_
#define poplibs_ExpressionGenerator_hpp_
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>

#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace popops {

// Interface wrapper for the below class.
void generateMappedOperations(
    const poplar::Target &target, const expr::Expr &expr,
    const std::vector<poplar::Tensor> &inputs,
    std::unordered_map<const expr::Expr *, poplar::Type> &constTypes,
    bool allInputsScalar, std::stringstream &outputCodeletCode);

poplar::Tensor generateAndExecuteMappedOperations(
    poplar::Graph &graph, const expr::Expr &expr,
    const std::vector<poplar::Tensor> &inputs,
    std::unordered_map<const expr::Expr *, poplar::Type> &constTypes,
    poplar::program::Sequence &prog, bool inPlace, bool allInputsScalar,
    const poplar::DebugNameAndId &dnai);

struct ExprInfo {
  bool isSupported;
  bool allInputsScalar;
};

struct InitializerStrings {
  std::string initializerString;
  std::string constantInitalizerStringScalar;
  std::string constantInitalizerStringVector;
};

ExprInfo analyseExpr(const expr::Expr &expr,
                     const std::vector<poplar::Tensor> &ins, bool isForcedOn);

// Traverses the expression tree and converts each into a string from the bottom
// up using Dijkstra's Two-Stack algorithm (using the call stack as the implicit
// second stack) and builds a C++ codelet from the expressions.
class GenerateCodeletFromMapExpr {
public:
  GenerateCodeletFromMapExpr(bool inPlace_,
                             const std::vector<poplar::Tensor> &ins)
      : data(), initalizers(), inputs(ins), numFusedOps(0),
        vectorizationIsSupported(true), inPlace(inPlace_){};

  // Traverse the expression tree and populate the data and initalizers fields.
  void traverseExpressionTree(
      const expr::Expr &expr,
      std::unordered_map<const expr::Expr *, poplar::Type> &constTypes);

  //  Create the codelet
  void generateCodelet(const poplar::Target &target, std::string &vertexName,
                       bool allInputsScalar, const expr::Expr &expr,
                       const InitializerStrings &initializerStrings,
                       bool isMultiVertex, std::stringstream &stream);

  // Create the codelet, save it to file, register the codelet to poplar, then
  // remove the file.
  std::string generateCodelet(poplar::Graph &graph, bool allInputsScalar,
                              const expr::Expr &expr,
                              const InitializerStrings &initializerStrings,
                              bool isMultiVertex);

  InitializerStrings generateInitializerStrings(const poplar::Target &target);

  poplar::Type deduceReturnType() const { return data.top().second; }

  bool isVectorized() const { return vectorizationIsSupported; }

  unsigned getVectorizationWidth(const poplar::Target &target) const;

  size_t getNumFusedOps() const { return numFusedOps; }

private:
  // Add the header section (includes, template traits, helper functions,
  // namespacing).
  void addHeader(std::stringstream &stream);
  // Add the footer section (namespacing).
  void addFooter(std::stringstream &stream);

  // Add a vectorized loop to the codelet.
  void addVectorizedSection(std::stringstream &stream,
                            size_t vectorizationWidth,
                            const std::string &initalizerString,
                            const std::string &constantInitalizerStringVector,
                            bool isMultivertex, unsigned numWorkers);

  // We always have non-vectorized serial equivalent. We always add this even if
  // we have a vectorized section as we may need to process a remainder as well.
  void addSerialSection(std::stringstream &stream,
                        const std::string &initalizerString,
                        const std::string &constantInitalizerString,
                        bool allInputsScalar, unsigned vectorizationWidth,
                        bool vectorizationIsSupported, bool isMultivertex,
                        unsigned numWorkers);

  // The string "data" which can be either a previously evaluated expression
  // (represented as a C++ variable name), a constant or a placeholder value.
  // We include its type as well for deducing the type of the next expression.
  using StringTypePair = std::pair<std::string, poplar::Type>;

  // At the end of the process this will just contain the variable name of the
  // final result. During the traversal it will contain the intermedate stack
  // of variable names/constants/placeholders which should be poped from upon
  // hitting an operation.
  std::stack<StringTypePair> data;

  // Each expression which is executed is converted to a string and stored as
  // an initalizer.
  std::queue<std::string> initalizers;

  // Each constant is added as its own initalizer. We can't do this lazily
  // because the function might be vectorized later. So we store the "const T
  // c1 = " part and then the actual constant as a string in a pair and
  // combine them later after we know what we are doing with vectorization.
  std::queue<std::pair<std::string, std::string>> constantInitalizers;

  // Hash a poplar::Type by its string representation.
  struct HashType {
    size_t operator()(const poplar::Type &t) const {
      return std::hash<std::string>()(t.toString());
    }
  };

  // Compare a poplar::Type by comparing the string representation.
  struct CompareType {
    bool operator()(const poplar::Type &lhs, const poplar::Type &rhs) const {
      return lhs.toString() == rhs.toString();
    }
  };
  // As we use types to the codelet we add them to this so they can be aliased
  // to a typedef for vectorization.
  std::unordered_set<poplar::Type, HashType, CompareType> TypesNeedingAlias;

  // Technically the user can provide more placeholders as input than they
  // actually use so we track which ones are used to avoid generating extra
  // loads.
  std::set<std::size_t> usedPlaceholders;

  const std::vector<poplar::Tensor> &inputs;

  // Number of operations we are fusing in this vertex.
  size_t numFusedOps;

  bool vectorizationIsSupported;

  bool inPlace;

public:
  static std::string createVertexName(const expr::Expr &expr,
                                      const std::vector<poplar::Tensor> &inputs,
                                      const bool inPlace,
                                      const bool allInputsScalar,
                                      const bool isMultiVertex);
};
} // namespace popops

#endif // poplibs_ExpressionGenerator_hpp_
