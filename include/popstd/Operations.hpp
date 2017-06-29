#ifndef __operations_hpp__
#define __operations_hpp__

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popstd {

poplar::Tensor add(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor abs(poplar::Graph &graph,
                   poplar::Tensor A,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor ceil(poplar::Graph &graph,
                    poplar::Tensor A,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

poplar::Tensor cos(poplar::Graph &graph,
                    poplar::Tensor A,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

poplar::Tensor div(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");


poplar::Tensor eq(poplar::Graph &graph,
                  poplar::Tensor A, poplar::Tensor B,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

poplar::Tensor exp(poplar::Graph &graph,
                   poplar::Tensor A,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor floor(poplar::Graph &graph,
                     poplar::Tensor A,
                     poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "");

poplar::Tensor gt(poplar::Graph &graph,
                  poplar::Tensor A, poplar::Tensor B,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");


poplar::Tensor gteq(poplar::Graph &graph,
                    poplar::Tensor A, poplar::Tensor B,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

poplar::Tensor lt(poplar::Graph &graph,
                  poplar::Tensor A, poplar::Tensor B,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");


poplar::Tensor lteq(poplar::Graph &graph,
                    poplar::Tensor A, poplar::Tensor B,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

poplar::Tensor log(poplar::Graph &graph,
                   poplar::Tensor A,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor logicalAnd(poplar::Graph &graph,
                          poplar::Tensor A, poplar::Tensor B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");

poplar::Tensor logicalNot(poplar::Graph &graph,
                          poplar::Tensor A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");

poplar::Tensor logicalOr(poplar::Graph &graph,
                         poplar::Tensor A, poplar::Tensor B,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "");

poplar::Tensor max(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor min(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor mul(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor neq(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor neg(poplar::Graph &graph,
                   poplar::Tensor A,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor pow(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor rem(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor signum(poplar::Graph &graph,
                      poplar::Tensor A,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

poplar::Tensor sub(poplar::Graph &graph,
                   poplar::Tensor A, poplar::Tensor B,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

poplar::Tensor tanh(poplar::Graph &graph,
                    poplar::Tensor A,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

poplar::Tensor select(poplar::Graph &graph,
                      poplar::Tensor A,
                      poplar::Tensor B,
                      poplar::Tensor pred,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

poplar::Tensor clamp(poplar::Graph &graph,
                      poplar::Tensor A,
                      poplar::Tensor lowerBound,
                      poplar::Tensor upperBound,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");
} // namespace popstd

#endif // __operations_hpp__
