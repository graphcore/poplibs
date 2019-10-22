#ifndef poplibs_PoolingDefUtil_hpp_
#define poplibs_PoolingDefUtil_hpp_
#include <poplibs_support/Compiler.hpp>
#include <popnn/PoolingDef.hpp>
#include <poputil/VertexTemplates.hpp>

// Specialize vertex template stringification for pooling type.
namespace poputil {

template <> struct VertexTemplateToString<popnn::PoolingType> {
  static std::string to_string(const popnn::PoolingType &op) {
    switch (op) {
    case popnn::PoolingType::MAX:
      return "popnn::PoolingType::MAX";
    case popnn::PoolingType::AVG:
      return "popnn::PoolingType::AVG";
    case popnn::PoolingType::SUM:
      return "popnn::PoolingType::SUM";
    }
    POPLIB_UNREACHABLE();
  }
};

} // end namespace poputil

#endif // poplibs_ExprOpUtil_hpp_
