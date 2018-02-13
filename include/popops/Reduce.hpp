#ifndef __popreduce_Reduce_hpp__
#define __popreduce_Reduce_hpp__

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popops {

/// Type of operation in a reduction
enum class Operation {
  ADD,
  MUL,
  MIN,
  MAX,
  AND,
  OR
};

/// Reduction with ADD operation on first dimension of tensor A
poplar::Tensor
reduce(poplar::Graph &graph, poplar::Tensor A,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix = "");


/// Reduction with ADD operation on first dimension of tensor B, scaled and
/// added to tensor B
///  A += k * reduction<ADD>(B)
void
reduceAcc(poplar::Graph &graph, poplar::Tensor A, float k,
          poplar::Tensor B,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "");

/// Perform a reduction over the first dimension of the partials tensor, writing
/// the result to the reduced tensor. The dimensions of the reduced tensor must
/// be the same as the dimensions of the partials tensor with the first
/// dimension removed. reduceMapping specifies a mapping from tiles to regions
/// of the reduced tensor that should be calculated on the tile.
void reduce(poplar::Graph &graph,
            poplar::Tensor partials,
            poplar::Tensor reduced,
            const std::vector<
              std::vector<poplar::Interval>
            > &reduceMapping,
            poplar::ComputeSet reduceCS);

/// Perform a reduction over the first dimension of the partials tensor, writing
/// the result to the reduced tensor. The dimensions of the reduced tensor must
/// be the same as the dimensions of the partials tensor with the first
/// dimension removed. reducedMapping specifies the tile mapping of the reduced
/// tensor. The mapping for the reduced tensor influences the mapping fo the
/// computation but it does not dictate it - elements may be reduced on
/// a different tile if that improves balance.
void reduceByDstMapping(poplar::Graph &graph,
                        poplar::Tensor partials,
                        poplar::Tensor reduced,
                        const std::vector<
                          std::vector<poplar::Interval>
                        > &reducedMapping,
                        poplar::ComputeSet reduceCS);


/// Perform  reduction<ADD>(A) * k with output tensor of type outTypeStr
poplar::Tensor reduceScale(poplar::Graph &graph, float k, poplar::Tensor &in,
                           const poplar::Type &outTypeStr,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "");

/// Perform a reduction operation of given type along the given dimensions of a
/// tensor. The relative order between the remaining dimensions in the input is
/// preserved in the output
/// \param graph
///        Graph to which reduction operation belongs to
/// \param A
///        Tensor to reduce
/// \param dims
///        Unordered dimensions to reduce (must be a subset of dimensions of A)
/// \param operation
///        The reduction operation (\see Operation)
/// \param prog
///        Poplar program to add the reduction to
/// \debugPrefix
///        String annotation
/// \return Tensor of rank rank(A) - size(dims) with the first dimension the
/// lowest dimension of A not part of dims
poplar::Tensor
reduce(poplar::Graph &graph, const poplar::Tensor &A,
       const std::vector<std::size_t> &dims,
       Operation operation,
       poplar::program::Sequence &prog,
       const std::string &debugPrefix = "");

}

#endif // __popreduce_Reduce_hpp__
