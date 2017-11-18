#ifndef __CircBuf_hpp__
#define __CircBuf_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <string>

namespace popstd {

class CircBuf {
  poplar::Graph &graph;
  unsigned size_;
  poplar::Tensor index;
  std::vector<std::size_t> shape;
  // The history buffer may be padded to ensure an integral number of grains
  unsigned padElements;
  poplar::Tensor hist;
public:
   CircBuf(poplar::Graph &graph, const poplar::Type &dataType,
           unsigned size, const std::vector<std::size_t> &shape,
           const std::string &debugPrefix = "");

   // return elements \a i entries old. i must be < \a size_
   poplar::Tensor prev(unsigned i, poplar::program::Sequence &seq,
                       const std::string &debugPrefix = "");

   // increment \a index and insert a new element
   void add(poplar::Tensor t, poplar::program::Sequence &seq,
            const std::string &debugPrefix = "");

   poplar::Tensor getIndex() const {
     return index;
   }

   unsigned size() const { return size_;}

};

} // namespace popstd
#endif // __CircBuf_hpp__
