// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ExpressionGenerator.hpp"
#include "ExprOpUtil.hpp"
#include "poplibs_support/Compiler.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/optional.hpp>

#include "ExprOpUtil.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <math.h>
#include <poplibs_support/logging.hpp>
#include <queue>
#include <sstream>
#include <stack>

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

using popops::expr::BinaryOpType;
using popops::expr::TernaryOpType;
using popops::expr::UnaryOpType;
using namespace poplibs_support;

namespace popops {

static bool isSupportedFpType(poplar::Type t) {
  return t == poplar::FLOAT || t == poplar::HALF;
}

static bool isSupportedSignedType(poplar::Type t) {
  return t == poplar::INT || t == poplar::LONGLONG;
}

static bool isSupportedUnsignedType(poplar::Type t) {
  return t == poplar::UNSIGNED_INT || t == poplar::BOOL ||
         t == poplar::UNSIGNED_LONGLONG;
}
static bool isSupportedType(poplar::Type t) {
  return isSupportedFpType(t) || isSupportedSignedType(t) ||
         isSupportedUnsignedType(t);
}

struct exprLog {
  uint32_t numberOfOperations = 0;
  std::vector<uint64_t> unintConstants;
  std::vector<int64_t> intConstants;
  std::vector<float> floatConstants;
  std::vector<BinaryOpType> binaryOps;
};

static bool traverseAndCheck(const expr::Expr &expr,
                             const std::vector<poplar::Tensor> &inputs,
                             exprLog &exprLog) {

  if (const expr::Const *c = expr.getAs<expr::Const>()) {

    poplar::Type type = c->getType();

    // Check if the type is floating point it is not NAN.
    if (type == poplar::FLOAT) {
      float val = *reinterpret_cast<float *>(c->getData());
      if (std::isinf(val) || std::isnan(val)) {
        return false;
      }
      exprLog.floatConstants.push_back(val);
    } else if (type == poplar::HALF) {
      assert(c->getTypeTraits().isFloat == true &&
             c->getTypeTraits().size == sizeof(float));

      float val = *reinterpret_cast<float *>(c->getData());
      if (std::isinf(val) || std::isnan(val)) {
        return false;
      }
      exprLog.floatConstants.push_back(val);
    } else if (isSupportedUnsignedType(type)) {
      uint64_t val;
      if (type == poplar::UNSIGNED_INT) {
        val =
            static_cast<uint64_t>(*reinterpret_cast<uint32_t *>(c->getData()));
      } else if (type == poplar::UNSIGNED_LONGLONG) {
        val = *reinterpret_cast<uint64_t *>(c->getData());

      } else if (type == poplar::BOOL) {
        val = static_cast<uint64_t>(*reinterpret_cast<bool *>(c->getData()));
      }
      exprLog.unintConstants.push_back(val);
    } else if (isSupportedSignedType(type)) {
      int64_t val;
      if (type == poplar::INT) {
        val = static_cast<int64_t>(*reinterpret_cast<int *>(c->getData()));
      } else if (type == poplar::LONGLONG) {
        val = *reinterpret_cast<int64_t *>(c->getData());
      }
      exprLog.intConstants.push_back(val);
    }

    return isSupportedType(c->getType());
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {

    // Check the placeholder type is supported.
    const size_t index = p->getIndex();

    if (index > inputs.size())
      return false;

    poplar::Type type = inputs[index - 1].elementType();
    return isSupportedType(type);
  } else if (const expr::Cast *c = expr.getAs<expr::Cast>()) {

    poplar::Type typeCastingTo = c->getRHSType();
    exprLog.numberOfOperations++;

    // Check the type being casted to is supported and also the type being
    // casted.
    return isSupportedType(typeCastingTo) &&
           traverseAndCheck(c->getLHS(), inputs, exprLog);

  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    exprLog.numberOfOperations++;

    return traverseAndCheck(u->getArg(), inputs, exprLog);
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {

    BinaryOpType opType = b->getOpType();
    if (opType == BinaryOpType::VARIANCE_TO_INV_STD_DEV ||

        opType == BinaryOpType::INV_STD_DEV_TO_VARIANCE) {
      return false;
    }
    exprLog.numberOfOperations++;
    exprLog.binaryOps.push_back(opType);
    if (!traverseAndCheck(b->getRHS(), inputs, exprLog) ||
        !traverseAndCheck(b->getLHS(), inputs, exprLog)) {
      return false;
    }

  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {

    exprLog.numberOfOperations++;
    if (!traverseAndCheck(t->getArg2(), inputs, exprLog) ||
        !traverseAndCheck(t->getArg1(), inputs, exprLog) ||
        !traverseAndCheck(t->getArg0(), inputs, exprLog)) {
      return false;
    }
  }

  return true;
}

static bool validateScalarOperations(const exprLog &exprLog) {
  // When all inputs are scalar there is no need for length information
  // in the vertex state. A fused vertex is optimised to do this, whereas
  // the elementwise codelets are not.
  // Also, when we generate fused vertices constants get absorbed into the
  // code rather than becoming tensors. So making a fused vertex is
  // generally beneficial, unless we generate many functions with specific
  // constants.
  // For example `add(x,const)` with different constants 1,2,3, 5.7 ...
  //
  // Scalar operations on 2 tensors are similarly better represented with a
  // fused vertex, but if there was one for every type and every operation
  // code space could grow.
  //
  // So limit the representation to a subset of operations and constant values
  // This is somewhat arbitrary but intended to cover likely loop increments
  // and other similar simple operations such as scaling results
  const std::vector<float> validFloatConstants = {-2.0f, -1.0f, -0.5f, 0.0f,
                                                  0.5f,  1.0f,  2.0f};
  const std::vector<uint64_t> validUintConstants = {0, 1, 2};
  const std::vector<int64_t> validIntConstants = {-2, -1, 0, 1, 2};

  using namespace popops::expr;
  const std::vector<BinaryOpType> validBinaryOps = {
      BinaryOpType::ADD, BinaryOpType::MULTIPLY, BinaryOpType::SUBTRACT,
      BinaryOpType::EQUAL};

  if (exprLog.unintConstants.size() + exprLog.intConstants.size() +
          exprLog.floatConstants.size() ==
      1) {
    // The expression contains a single constant, limit use to a number of
    // allowed values depending on type
    unsigned mismatches = 0;
    for (const auto val : exprLog.floatConstants) {
      mismatches +=
          std::find(validFloatConstants.begin(), validFloatConstants.end(),
                    val) == validFloatConstants.end();
    }
    for (const auto val : exprLog.unintConstants) {
      mismatches +=
          std::find(validUintConstants.begin(), validUintConstants.end(),
                    val) == validUintConstants.end();
    }
    for (const auto val : exprLog.intConstants) {
      mismatches +=
          std::find(validIntConstants.begin(), validIntConstants.end(), val) ==
          validIntConstants.end();
    }
    return mismatches == 0;
  } else if (exprLog.binaryOps.size() == 1) {
    // The expression contains a single binary op and therefore 2 tensors,
    // limit use depending on operation
    unsigned mismatches = 0;
    for (const auto op : exprLog.binaryOps) {
      mismatches += std::find(validBinaryOps.begin(), validBinaryOps.end(),
                              op) == validBinaryOps.end();
    }
    return mismatches == 0;
  }
  return false;
}

ExprInfo analyseExpr(const expr::Expr &expr,
                     const std::vector<poplar::Tensor> &inputs,
                     bool isForcedOn) {
  if (inputs.size() == 0)
    return {false, false};

  // All tensors should be the same shape or scalar
  unsigned size = 1;
  auto shape = inputs[0].shape();
  for (const Tensor &t : inputs) {
    if (size == 1) {
      size = t.numElements();
      shape = (size == 1) ? t.flatten().shape() : t.shape();
    }
    if ((t.shape() != shape && t.numElements() != 1) || t.containsAliases()) {
      return {false, false};
    }
  }
  exprLog exprLog;
  bool isOk = traverseAndCheck(expr, inputs, exprLog);

  // Check that this is not just a single operation.
  isOk &= isForcedOn || exprLog.numberOfOperations > 1;
  if (size == 1 && exprLog.numberOfOperations == 1) {
    // If it is a single operation using only scalar values some cases are
    // still worthwhile making a codelet
    isOk |= validateScalarOperations(exprLog);
  }
  return {isOk, (size == 1)};
}

namespace {

std::string getTypeAlias(const std::string &typeAsStr) {
  std::string str = typeAsStr + "_ty";
  size_t pos = str.find("unsigned");
  if (pos != std::string::npos) {
    str.replace(0, 9, "u", 1);
  }
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  return str;
}

void executeCodelet(Graph &graph, const std::string &codeletName,
                    const boost::optional<std::string> &multiVertexCodeletName,
                    std::vector<Tensor> inputs, const Tensor &out,
                    const std::vector<std::vector<Interval>> &intervals,
                    unsigned tile, const ComputeSet &cs, size_t numFusedOps,
                    bool inPlace) {
  const auto dType = inputs[0].elementType();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto grainSize =
      std::max<unsigned>(vectorWidth, target.getAtomicStoreGranularity());
  const auto numWorkers = target.getNumWorkerContexts();

  // When the alternative is a 1D single worker vertex it's not worth using
  // a Multivertex
  const bool inputIsSmallAndContiguous = intervals.size() == 1 &&
                                         intervals[0].size() == 1 &&
                                         intervals[0][0].size() <= vectorWidth;

  const bool isMultiVertex = multiVertexCodeletName && intervals.size() == 1 &&
                             !inputIsSmallAndContiguous;

  const auto vertexRegions = [&]() {
    if (isMultiVertex) {
      std::vector<std::vector<std::vector<poplar::Interval>>> result(1);
      result[0].resize(1);
      result[0][0] = intervals[0];
      logging::popops::trace("Creating 1 MultiVertex on tile {}, name {}", tile,
                             multiVertexCodeletName.get());
      return result;
    }
    auto result =
        splitRegionsBetweenWorkers(target, intervals, grainSize, 2 * grainSize);
    if (intervals.size()) {
      logging::popops::trace("Creating {} Vertices on tile {}, name {}",
                             result.size(), tile, codeletName);
    }
    return result;
  }();
  for (const auto &regions : vertexRegions) {
    auto v = graph.addVertex(cs, isMultiVertex ? multiVertexCodeletName.get()
                                               : codeletName);

    poplar::Tensor outRegions = poplar::concat(out.flatten().slices(regions));

    std::vector<poplar::Tensor> inRegions(inputs.size());

    std::transform(inputs.begin(), inputs.end(), inRegions.begin(),
                   [&](poplar::Tensor &t) {
                     return poplar::concat(t.flatten().slices(regions));
                   });

    std::uint64_t estimate = isMultiVertex ? 20 : 13;
    for (unsigned i = 0; i < inRegions.size(); ++i) {
      if (inputs[i].numElements() == 1) {

        graph.connect(v["in" + std::to_string(i + 1)], inputs[i].reshape({}));
      } else {
        graph.connect(v["in" + std::to_string(i + 1)], inRegions[i]);
      }

      estimate += inRegions[i].numElements() / vectorWidth * numFusedOps;
      estimate += inRegions[i].numElements() % vectorWidth * numFusedOps;
    }
    if (isMultiVertex) {
      estimate = gccs::ceildiv(estimate, numWorkers) * numWorkers;
    }
    graph.setPerfEstimate(v, estimate);

    if (!inPlace) {
      graph.connect(v["out"],
                    out.numElements() == 1 ? out.reshape({}) : outRegions);
    }
    graph.setTileMapping(v, tile);
  }
}

} // end anonymous namespace

// Top level function which calls functions to generate and execute the
// generated codelet.
poplar::Tensor generateAndExecuteMappedOperations(
    Graph &graph, const expr::Expr &expr, const std::vector<Tensor> &inputs,
    std::unordered_map<const expr::Expr *, Type> &constTypes, Sequence &prog,
    bool inPlace, bool allInputsScalar, const DebugNameAndId &dnai) {

  GenerateCodeletFromMapExpr generate{inPlace, inputs};

  // Traverse the expression tree and based on each node in the tree build up
  // the body of the map operation in a string format representing the end code.
  generate.traverseExpressionTree(expr, constTypes);

  poplar::Type returnType = generate.deduceReturnType();

  // Generate the actual codelet (single worker 1D) and a MultiVertex.
  // Both or just one of these may actually be executed depending on if each
  // tile contains > 1 region or not.
  // However both are compiled, added to the graph, and the names of the
  // generated codelets are stored in codeletName, codeletNameMultiVertex.
  const auto initialiserStrings = generate.generateInitializerStrings(graph);
  const auto codeletName = generate.generateCodelet(
      graph, allInputsScalar, expr, initialiserStrings, false);

  size_t numFusedOp = generate.getNumFusedOps();

  const auto vectorizationWidth = generate.getVectorizationWidth(graph);

  std::vector<Tensor> flattenedIns;
  std::vector<Tensor *> asPtr;
  std::vector<Tensor> vectorIns;

  flattenedIns.reserve(inputs.size());
  vectorIns.reserve(inputs.size());

  // Flatten the input and also record the address of the flattened tensor if it
  // has > 1 element.
  for (unsigned i = 0; i < inputs.size(); ++i) {
    flattenedIns.push_back(inputs[i].flatten());
    if (inputs[i].numElements() != 1) {
      asPtr.push_back(&flattenedIns[i]);
      vectorIns.push_back(inputs[i]);
    }
  }

  poplar::Tensor out;

  if (inPlace) {
    out = inputs[0];
  } else if (allInputsScalar) {
    // All tensors have 1 element, the output will be the tensor with the
    // highest rank
    // Eg from {1,1,1}, {1} the result is {1,1,1}
    const auto inputToClone = *std::max_element(
        inputs.begin(), inputs.end(),
        [](const Tensor &a, const Tensor &b) { return a.rank() < b.rank(); });
    out = graph.clone(returnType, inputToClone, {dnai, codeletName + "/Out"});
  } else {
    out = createOutputForElementWiseOp(
        graph, vectorIns.size() == 0 ? inputs : vectorIns, returnType,
        {dnai, codeletName + "/Out"});
  }

  const auto codeletNameMultiVertex = [&]() {
    // Check if each worker in a multivertex can write without conflicting with
    // other workers
    auto target = graph.getTarget();
    const auto outputWritesAreAtomic =
        target.getTypeSize(out.elementType()) * vectorizationWidth >=
        target.getAtomicStoreGranularity();

    const auto vectorizedMultivertex =
        generate.isVectorized() && vectorizationWidth > 1;
    const auto nonVectorizedMultivertex =
        !generate.isVectorized() && vectorizationWidth == 1;

    // Only make a multivertex where we can do so
    boost::optional<std::string> result;
    if (outputWritesAreAtomic &&
        (vectorizedMultivertex || nonVectorizedMultivertex)) {
      result = generate.generateCodelet(graph, allInputsScalar, expr,
                                        initialiserStrings, true);
    }
    return result;
  }();

  auto outFlat = out.flatten();
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai});
  graph.reorderToSimplify(&outFlat, asPtr, false);
  const auto mapping = graph.getTileMapping(outFlat);
  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    executeCodelet(graph, codeletName, codeletNameMultiVertex, flattenedIns,
                   outFlat, tileContiguousRegions, tile, cs, numFusedOp,
                   inPlace);
  }
  prog.add(Execute(cs, {dnai}));

  return out;
}

// Convert a constant expression into a string representing that constant in
// C/C++.
static std::string handleConstant(const expr::Const *c) {
  return c->printValue();
}

static bool typeSupportsVectorization(poplar::Type type) {
  return type == poplar::HALF || type == poplar::FLOAT || type == poplar::BOOL;
}

void GenerateCodeletFromMapExpr::traverseExpressionTree(
    const expr::Expr &expr,
    std::unordered_map<const expr::Expr *, Type> &constTypes) {

  if (const expr::Const *c = expr.getAs<expr::Const>()) {

    const poplar::Type type = constTypes.at(c);
    TypesNeedingAlias.insert(type);

    std::string typeAsStr = getTypeAlias(type.toString());
    const std::string constantAsString = handleConstant(c);
    const std::string variableName =
        "C" + std::to_string(constantInitalizers.size() + 1);
    const std::string initalizer =
        "const " + typeAsStr + " " + variableName + " = ";

    vectorizationIsSupported &= typeSupportsVectorization(type);

    constantInitalizers.push({initalizer, constantAsString});
    data.push({variableName, type});

  } else if (const expr::Cast *c = expr.getAs<expr::Cast>()) {
    traverseExpressionTree(c->getLHS(), constTypes);

    poplar::Type typeCastingTo = c->getRHSType();
    auto pair = data.top();
    data.pop();

    TypesNeedingAlias.insert(typeCastingTo);

    // Propagate the fact that the operand is a constant.
    std::string variable_name = "my_var_" + std::to_string(initalizers.size());

    if (pair.first[0] == 'C') {
      variable_name.insert(variable_name.begin(), 'C');
    }
    std::string asStr = getTypeAlias(typeCastingTo.toString());
    std::string result = "const " + asStr + " " + variable_name + " = (" +
                         asStr + ")" + pair.first + ";";

    vectorizationIsSupported = false;

    // The initializer to be printed in the function.
    data.push({variable_name, typeCastingTo});
    // The variable name to be used in subsequent iterations.
    initalizers.push(result);
  } else if (const expr::PlaceHolder *p = expr.getAs<expr::PlaceHolder>()) {
    const size_t index = p->getIndex();

    const std::string placeholder = "load" + std::to_string(index);

    poplar::Type type = inputs[index - 1].elementType();

    vectorizationIsSupported &= typeSupportsVectorization(type);

    data.push({placeholder, type});
    TypesNeedingAlias.insert(type);
    usedPlaceholders.insert(index);
  } else if (const expr::UnaryOp *u = expr.getAs<expr::UnaryOp>()) {
    numFusedOps++;
    traverseExpressionTree(u->getArg(), constTypes);

    assert(!data.empty() &&
           "Expression traversal failed in unary op, data is empty");

    auto opType = u->getOpType();
    auto pair = data.top();
    data.pop();

    const std::string &param = pair.first;

    std::string variable_name = "my_var_" + std::to_string(initalizers.size());

    // Propagate the fact that the operand is a constant.
    if (pair.first[0] == 'C') {
      variable_name.insert(variable_name.begin(), 'C');
    }
    poplar::Type type = getReturnType(opType, pair.second);

    TypesNeedingAlias.insert(type);

    std::string asStr = getTypeAlias(type.toString());

    std::string result = "const " + asStr + " " + variable_name + " = ";
    if (isSpecialCase(opType)) {
      result += handleSpecialCase(opType, param);
    } else {
      result += getUnaryOpAsString(opType, type);
      result += "(" + param + ")";
    }
    result += ";\n";

    vectorizationIsSupported &= supportsVectorization(opType);
    // The initializer to be printed in the function.
    data.push({variable_name, type});
    // The variable name to be used in subsequent iterations.
    initalizers.push(result);
  } else if (const expr::BinaryOp *b = expr.getAs<expr::BinaryOp>()) {
    numFusedOps++;
    auto opType = b->getOpType();

    traverseExpressionTree(b->getRHS(), constTypes);
    traverseExpressionTree(b->getLHS(), constTypes);

    assert(data.size() >= 2 &&
           "Expression traversal failed in binary op, data is less than 2");

    auto pair1 = data.top();
    data.pop();
    auto pair2 = data.top();
    data.pop();

    const std::string &param1 = pair1.first;
    const std::string &param2 = pair2.first;

    std::string variable_name = "my_var_" + std::to_string(initalizers.size());
    // Propagate the fact that the operand is a constant.
    if (pair1.first[0] == 'C' && pair2.first[0] == 'C') {
      variable_name.insert(variable_name.begin(), 'C');
    }
    poplar::Type type = getReturnType(opType, pair1, pair2);

    TypesNeedingAlias.insert(type);

    std::string result =
        "const " + getTypeAlias(type.toString()) + " " + variable_name + " = ";

    if (hasFunctionSemantics(opType)) {
      // Call it like a function.
      result += std::string(getBinaryOpAsString(opType, type)) + "(" + param1 +
                "," + param2 + ")";
    } else if (isSpecialCase(opType)) {
      result += handleSpecialCase(opType, param1, param2);
    } else {
      result += param1;
      result += getBinaryOpAsString(opType, type);
      result += param2;
    }

    result += ";\n";
    // The initializer to be printed in the function.
    data.push({variable_name, type});
    // The variable name to be used in subsequent iterations.
    initalizers.push(result);
  } else if (const expr::TernaryOp *t = expr.getAs<expr::TernaryOp>()) {
    numFusedOps++;
    auto opType = t->getOpType();

    traverseExpressionTree(t->getArg2(), constTypes);
    traverseExpressionTree(t->getArg1(), constTypes);
    traverseExpressionTree(t->getArg0(), constTypes);

    assert(data.size() >= 3 &&
           "Expression traversal failed in ternary op, data is less than 2");

    // Pop the three arguments from the stack.
    auto pair1 = data.top();
    data.pop();
    auto pair2 = data.top();
    data.pop();
    auto pair3 = data.top();
    data.pop();

    bool isFloatingPoint =
        pair2.second == poplar::HALF || pair2.second == poplar::FLOAT;

    const std::string variable_name =
        "my_var_" + std::to_string(initalizers.size());
    std::string result;

    // Select is implemented as if (C) A; else B;
    if (opType == TernaryOpType::SELECT) {
      TypesNeedingAlias.insert(pair1.second);
      poplar::Type returnType = pair1.second;

      bool lhsIsConst = pair1.first[0] == 'C';
      if (lhsIsConst) {
        returnType = pair2.second;
      }

      const std::string type = getTypeAlias(returnType.toString());
      result = type + " " + variable_name + ";";

      result += "if (" + pair3.first + ") { " + variable_name + " = " +
                pair1.first + ";} else {" + variable_name + " = " +
                pair2.first + ";}";

    } else {
      assert(opType == TernaryOpType::CLAMP &&
             "TernaryOpType is not supported by expression generator.");
      const std::string maxFunc = isFloatingPoint ? "NAMESPACE::fmax" : "max";
      const std::string minFunc = isFloatingPoint ? "NAMESPACE::fmin" : "min";

      TypesNeedingAlias.insert(pair2.second);
      const std::string type = getTypeAlias(pair2.second.toString());

      // Clamp is 'const Type VAR = max(low, min(val, high));'
      result = "const " + type + " " + variable_name + " = " + maxFunc + "(" +
               pair2.first + "," + minFunc + "(" + pair1.first + "," +
               pair3.first + "));";
    }

    // The initializer to be printed in the function.
    data.push({variable_name, pair2.second});
    // The variable name to be used in subsequent iterations.
    initalizers.push(result);
  }
}

// Generate the actual codelet.

void GenerateCodeletFromMapExpr::addHeader(std::stringstream &stream) {
  stream << R"l(
#include <poplar/HalfFloat.hpp>
  #ifdef __IPU__

  // Use the IPU intrinsics
  #include <ipu_memory_intrinsics>
  #include <ipu_vector_math>
  #define NAMESPACE ipu
  #else
  // Use the std functions
  #include <cmath>
  #define NAMESPACE std
  #endif
template <typename T>
const T &max(const T &x, const T &y) {
  return x < y ? y : x;
}

template <typename T>
const T &min(const T &x, const T &y) {
  return x < y ? x : y;
}

  template<typename T>
  struct Traits {
  typename std::remove_reference<T>::type ONE() { return 1; }
  };

  template<>
  struct Traits<double> { static double ONE() { return 1.0;} };

  template<>
  struct Traits<double&> { static double ONE() { return 1.0;} };

  template<>
  struct Traits<const double&> { static double ONE() { return 1.0;} };

  template<>
  struct Traits<float> { static float ONE(){ return 1.0f;} };

  template<>
  struct Traits<float&> { static float ONE() {return 1.0f;} };

  template<>
  struct Traits<const float&> { static float ONE() {return 1.0f;} };


  template<>
  struct Traits<half> { static half ONE() {return 1;} };

  template<>
  struct Traits<half&> { static half ONE() {return 1;} };

#ifdef __IPU__
  template<>
  struct Traits<float2> { static float2 ONE() { return {1.0f, 1.0f};} };
  template<>
  struct Traits<float2&> { static float2 ONE() { return {1.0f, 1.0f};}  };
  template<>
  struct Traits<const float2&> { static float2 ONE() { return {1.0f, 1.0f};}  };

  template<>
  struct Traits<half2> { static half2 ONE() { return {1.0, 1.0};} };
  template<>
  struct Traits<half2&> { static half2 ONE() {return {1.0, 1.0};}  };
  template<>
  struct Traits<const half2&> { static half2 ONE() {return {1.0, 1.0};}  };

  template<>
  struct Traits<half4> { static half4 ONE(){return {1.0, 1.0,1.0, 1.0};}  };
  template<>
  struct Traits<half4&> { static half4 ONE(){return {1.0, 1.0, 1.0, 1.0};}  };
  template<>
  struct Traits<const half4&> { static half4 ONE(){return {1.0, 1.0, 1.0, 1.0};}  };
#endif

  template<typename T>
  inline T internal_rsqrt(T x) {
  #ifdef __IPU__
      return ipu::rsqrt(x);
  #else
     return Traits<T>::ONE() / std::sqrt(x);
  #endif
  }

  template <typename T>
  inline T internal_remainder(T x, T y) {
    if (std::is_integral<T>::value) {
        T tmp = x / y;
        return x - tmp*y;
    } else {
        return NAMESPACE::fmod(float(x), float(y));
    }
  }


 template <typename T>
  inline T internal_sigmoid(T x) {
    #ifdef __IPU__
      return ipu::sigmoid(x);
    #else
      T one = Traits<T>::ONE();
      return one / (one + NAMESPACE::exp(-x));
    #endif
  }

  #include <poplar/Vertex.hpp>
  using namespace poplar;

  namespace popops {
  namespace map {
  )l";
}

void GenerateCodeletFromMapExpr::addFooter(std::stringstream &stream) {
  stream << R"l(
  } // end namespace map
  } // end namespace popops
  )l";
}

std::string loopCountString(const std::string &sizeStr,
                            unsigned vectorizationWidth, unsigned numWorkers,
                            bool isMultiVertex) {
  auto log2VectorWidth = gccs::ceilLog2(vectorizationWidth);
  if (!isMultiVertex) {
    return "unsigned loopCount = " + sizeStr + ".size() >>" +
           std::to_string(log2VectorWidth) + ";";
  }
  if (vectorizationWidth == 1) {
    return "unsigned loopCount = ((" + sizeStr + "+" +
           std::to_string(numWorkers - 1) + "-wid)*0xaaab)>>18;\n";
  }
  return "unsigned loopCount = (((" + sizeStr + ".size() >>" +
         std::to_string(log2VectorWidth) + ")+" +
         std::to_string(numWorkers - 1) + "-wid)*0xaaab)>>18;\n";
}

// Add a vectorized loop to the codelet.
void GenerateCodeletFromMapExpr::addVectorizedSection(
    std::stringstream &stream, size_t vectorizationWidth,
    const std::string &initalizerString,
    const std::string &constantInitalizerString, bool isMultiVertex,
    unsigned numWorkers) {

  stream << R"l(// Vectorized code
            #ifdef __IPU__
            {)l";

  for (poplar::Type type : TypesNeedingAlias) {

    stream << "using " << getTypeAlias(type.toString()) << " = "
           << type.toString() << std::to_string(vectorizationWidth) << ";\n";
  }

  // MultiVertex initial offset and stride
  if (isMultiVertex) {
    stream << "const unsigned workerOffset = wid * "
           << std::to_string(vectorizationWidth) << ";\n";
    stream << "constexpr unsigned stride = " << numWorkers << ";\n";
  } else {
    stream << "constexpr unsigned workerOffset = 0;\n";
    stream << "constexpr unsigned stride = 1;\n";
  }

  // Add each input as a pointer cast.
  for (std::size_t index : usedPlaceholders) {
    const std::string type =
        getTypeAlias(inputs[index - 1].elementType().toString());
    const std::string id = std::to_string(index);
    // Add: "const {type} * In{id} = reinterpret_cast<{type}*>(in{id});"
    if (inputs[index - 1].numElements() != 1) {
      stream << "const " << type << " * In" << id
             << " = reinterpret_cast<const " << type << "*>(&in" << id
             << "[workerOffset]);\n";
    }
  }

  assert(!data.empty() && "Attempting to read data stack which is empty");
  const std::string outType = getTypeAlias(data.top().second.toString());

  const std::string outString = inPlace ? "in1" : "out";

  // Add: "{outType} * In{id} = reinterpret_cast<{type}*>({in1/out});"
  stream << outType << " * Out "
         << " = reinterpret_cast<" << outType << "*>(&" << outString
         << "[workerOffset]);\n";

  stream << "remainder = " << outString << " .size() %"
         << std::to_string(vectorizationWidth)
         << ";\nstartIndex = " << outString << ".size() - remainder;\n";

  stream << loopCountString(outString, vectorizationWidth, numWorkers,
                            isMultiVertex);
  stream << R"l(
            while (loopCount) {
              loopCount--;
              )l";

  // Load the data.
  for (std::size_t index : usedPlaceholders) {
    const std::string type =
        getTypeAlias(inputs[index - 1].elementType().toString());
    const std::string id = std::to_string(index);

    // Add: load{id} = ipu::load_postinc(&In{id}, stride);
    if (inputs[index - 1].numElements() != 1) {
      stream << type << " load" + id << "= ipu::load_postinc(&In" << id
             << ", stride);\n";
    }
  }

  stream << constantInitalizerString;
  // Each expression is a variable initialization.
  stream << initalizerString;

  assert(!data.empty() && "Attempting to read data stack which is empty");
  // Add: "ipu::store_postinc(&Out, {result}, stride);"
  stream << "ipu::store_postinc(&Out," << data.top().first << ",stride);\n";
  stream << R"l(
        } // End loop
        } // End vectorized section.
        #endif)l";
}

// Adds the serial section of the codelet to the stream.
void GenerateCodeletFromMapExpr::addSerialSection(
    std::stringstream &stream, const std::string &initalizerString,
    const std::string &constantInitalizerString, bool allInputsScalar,
    unsigned vectorizationWidth, bool vectorizationIsSupported,
    bool isMultiVertex, unsigned numWorkers) {

  stream << R"l(
        // Remainder/Serial fallback.
        {)l";

  for (poplar::Type type : TypesNeedingAlias) {

    std::string asStr = getTypeAlias(type.toString());

    stream << "using " << asStr << " = " << type.toString() << ";\n";
  }
  const bool doRemainder = isMultiVertex && (vectorizationWidth > 1);
  if (doRemainder) {
    // This function is generating code to deal with the remainder, and
    // only the last worker deals with the remainder
    stream << R"l(
          if(wid == )l"
           << numWorkers - 1 << R"l() {
      )l";
  }
  const bool nonVectorizedMultivertex =
      vectorizationWidth == 1 && isMultiVertex;
  if (nonVectorizedMultivertex) {
    // This function is generating the code to deal with a non vectorised type
    // and it is a multivertex
    stream << loopCountString("remainder", vectorizationWidth, numWorkers,
                              isMultiVertex);
    stream << R"l(
          unsigned i = wid;
          while (loopCount) {
            loopCount--;

      )l";
  } else if (vectorizationWidth == 2 && vectorizationIsSupported) {
    // If the vectorization width = 2 there can only be 1 remainder so we don't
    // need a loop to deal with it.
    // However if !IPUModel then this is a serial, scalar loop instead which
    // deals with all the data.
    stream << R"l(
          #ifdef __IPU__
            if(remainder) {
              unsigned i = startIndex;
          #else
              for (unsigned i = startIndex; i < startIndex + remainder; ++i) {
          #endif
      )l";
  } else {
    // Loop over the remainder - case of vectorization width > 2
    stream << R"l(
          for (unsigned i = startIndex; i < startIndex + remainder; ++i) {
      )l";
  }
  // Add the aliases to the "load" variable names which the placeholders are
  // using.
  for (std::size_t index : usedPlaceholders) {
    std::string type = getTypeAlias(inputs[index - 1].elementType().toString());
    const std::string id = std::to_string(index);

    if (!inPlace || index != 1) {
      // Use const declaration for all inputs with the following exception.
      // If `inPlace` flag is set, the input with index `1` is an Input/Output
      // field, so is not const declared.
      stream << "const ";
    }

    // Add: "{type} & load{id} = in{id}[i];"
    if (inputs[index - 1].numElements() == 1) {
      stream << type << " load" << id << " =  in" << id << ";\n";
    } else {
      stream << type << "& load" << id << " =  in" << id << "[i];\n";
    }
  }

  // Add the constants.
  stream << constantInitalizerString;

  // Add the variable initalizations that make up the expression.
  stream << initalizerString;

  // The final assignment of the aggregate of all the operations in
  // initalizers.
  if (allInputsScalar) {
    stream << (inPlace ? "*in1 = " : "*out = ");
  } else {
    stream << (inPlace ? "in1[i] = " : "out[i] = ");
  }

  assert(!data.empty() && "Attempting to read data stack which is empty");
  stream << data.top().first << ";\n";

  if (nonVectorizedMultivertex) {
    stream << "i+=" << numWorkers << ";\n";
  }
}

InitializerStrings GenerateCodeletFromMapExpr::generateInitializerStrings(
    const poplar::Graph &graph) {
  // Each stage of the operation is stored as a variable initalization.
  InitializerStrings result;

  while (!initalizers.empty()) {
    result.initializerString += initalizers.front();
    initalizers.pop();
  }

  const auto vectorizationWidth = getVectorizationWidth(graph);
  // Process the constant values. We need this step as we cannot just embed
  // the constants if we are working with vectors.
  while (!constantInitalizers.empty()) {

    auto &pair = constantInitalizers.front();

    // Just output the constant as "const T C1 = CONST;"
    result.constantInitalizerStringScalar += pair.first + pair.second + ";\n";

    // Turn the constant into a vector. I.E for vector size of 2: "const T C1
    // = {CONST, CONST};"
    result.constantInitalizerStringVector += pair.first + "{";
    for (unsigned i = 0; i < vectorizationWidth; ++i) {
      result.constantInitalizerStringVector += pair.second;
      if (i != vectorizationWidth - 1) {
        result.constantInitalizerStringVector += ", ";
      }
    }
    result.constantInitalizerStringVector += "};\n";
    constantInitalizers.pop();
  }
  // Create vectorised versions of all the scalar Tensors
  for (unsigned i = 0; i < inputs.size(); i++) {
    if (inputs[i].numElements() == 1) {
      const std::string type = getTypeAlias(inputs[i].elementType().toString());
      result.constantInitalizerStringVector +=
          type + " load" + std::to_string(i + 1) + "={";
      for (unsigned j = 0; j < vectorizationWidth; j++) {
        result.constantInitalizerStringVector +=
            "in" + std::to_string(i + 1) +
            (j != vectorizationWidth - 1 ? "," : "};\n");
      }
    }
  }
  return result;
}

std::string GenerateCodeletFromMapExpr::generateCodelet(
    poplar::Graph &graph, bool allInputsScalar, const expr::Expr &expr,
    const InitializerStrings &initializerStrings, bool isMultiVertex) {

  const auto vectorizationWidth = getVectorizationWidth(graph);
  const std::string vertexName =
      createVertexName(expr, inputs, inPlace, allInputsScalar, isMultiVertex);

  const std::string namespacedVertexName = "popops::map::" + vertexName;

  if (graph.hasCodelet(namespacedVertexName)) {
    logging::popops::debug("Codelet already in graph {}", namespacedVertexName);
    return namespacedVertexName;
  }

  std::stringstream stream;
  std::stringstream body_stream;

  addHeader(stream);

  stream << R"l(
  class )l";

  stream << vertexName << " : public";
  stream << (isMultiVertex ? " MultiVertex" : " Vertex");
  stream << "{\npublic:\n";

  // Constructor.
  stream << vertexName << "();\n";

  // The output. Aligned to 8 to support vectorization.
  if (!inPlace) {
    assert(!data.empty() && "Attempting to read data stack which is empty");
    if (allInputsScalar) {
      body_stream << "Output<" << data.top().second.toString() << "> out;\n";
    } else {
      body_stream << "Output<Vector<" << data.top().second.toString()
                  << ",VectorLayout::SPAN, 8 >> out;\n";
    }
  }

  // The inputs/inplace outputs. Aligned to 8 for vectorization. We generate
  // these even if they are unused as the part that runs the code doesn't have
  // the information on which inputs are used or not.
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (i == 0 && inPlace) {
      if (allInputsScalar) {
        body_stream << "InOut<" << inputs[i].elementType().toString() << ">";
      } else {
        body_stream << "InOut<Vector<" << inputs[i].elementType().toString()
                    << ",VectorLayout::SPAN, 8 >>";
      }
    } else {
      if (inputs[i].numElements() == 1) {
        body_stream << "Input<" << inputs[i].elementType().toString() << ">";
      } else {
        body_stream << "Input<Vector<" << inputs[i].elementType().toString()
                    << ", VectorLayout::ONE_PTR, 8>>";
      }
    }
    body_stream << " in" << std::to_string(i + 1) << ";\n";
  }

  // Add the start of the actual compute function.
  if (isMultiVertex) {
    body_stream << R"l(
            bool compute(unsigned wid) {)l";
  } else {
    body_stream << R"l(
            bool compute() {)l";
  }

  // If we are vectorizing we will need a serial section to calculate the
  // remainder if the vectorization amount doesn't divide evenly.
  if (allInputsScalar) {
    body_stream << R"l(
          unsigned startIndex = 0;
          unsigned remainder = 1;
        )l";
  } else {
    if (inPlace) {
      body_stream << R"l(
          unsigned startIndex = 0;
          unsigned remainder = in1.size();
        )l";
    } else {
      body_stream << R"l(
              unsigned startIndex = 0;
              unsigned remainder = out.size();)l";
    }
  }
  const auto numWorkers = graph.getTarget().getNumWorkerContexts();
  // If we can generate a vectorized version add it to the codelet.
  if (vectorizationIsSupported && vectorizationWidth > 1 && !allInputsScalar) {
    addVectorizedSection(body_stream, vectorizationWidth,
                         initializerStrings.initializerString,
                         initializerStrings.constantInitalizerStringVector,
                         isMultiVertex, numWorkers);
  }

  addSerialSection(body_stream, initializerStrings.initializerString,
                   initializerStrings.constantInitalizerStringScalar,
                   allInputsScalar, vectorizationWidth,
                   vectorizationIsSupported, isMultiVertex, numWorkers);

  stream << body_stream.str();
  if (isMultiVertex && vectorizationWidth > 1) {
    stream << " } // End if (wid == numWorkers-1) ";
  }
  stream << R"l(
          }  // End loop
        }// End serial version.
      return true;
      }
    };
  )l";

  addFooter(stream);
  logging::popops::debug("Adding codelet {} to graph", namespacedVertexName);
  graph.addCodelets(stream);

  return namespacedVertexName;
}

std::string GenerateCodeletFromMapExpr::createVertexName(
    const expr::Expr &expr, const std::vector<poplar::Tensor> &inputs,
    const bool inPlace, const bool allInputsScalar, const bool isMultiVertex) {
  std::string result = expr.name(inputs);
  result += std::to_string(inPlace);
  result += std::to_string(allInputsScalar);
  for (const auto &input : inputs) {
    result += std::to_string(input.numElements() == 1);
  }
  if (isMultiVertex) {
    result += "_MultiVertex";
  }
  return result;
}

unsigned
GenerateCodeletFromMapExpr::getVectorizationWidth(const Graph &graph) const {
  const auto &target = graph.getTarget();
  // Get the smallest vectorization width of all the types.
  poplar::Type smallestVector = *std::min_element(
      TypesNeedingAlias.begin(), TypesNeedingAlias.end(),
      [&target](const poplar::Type &lhs, const poplar::Type &rhs) {
        return target.getVectorWidth(lhs) < target.getVectorWidth(rhs);
      });

  const auto smallestVectorPer64Bits = 8u / target.getTypeSize(smallestVector);
  const auto vectorizationWidth = std::min<std::size_t>(
      target.getVectorWidth(smallestVector), smallestVectorPer64Bits);
  return vectorizationWidth;
}

} // namespace popops
