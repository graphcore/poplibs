#ifndef __History_hpp__
#define __History_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <string>

namespace popstd {

class History {
  poplar::Graph &graph;
  unsigned size_;
  poplar::Tensor index;
  std::vector<std::size_t> shape;
  poplar::Tensor hist;
public:
   History(poplar::Graph &graph, const std::string &dataType,
           unsigned size, const std::vector<std::size_t> &shape);

   poplar::Tensor prev(unsigned i, poplar::program::Sequence &seq,
                       const std::string &debugPrefix = "");

   void add(poplar::Tensor t, poplar::program::Sequence &seq,
            const std::string &debugPrefix = "");

   poplar::Tensor getIndex() const {
     return index;
   }

   unsigned size() const { return size_;}

};

} // namespace popstd
#endif // __History_hpp__
