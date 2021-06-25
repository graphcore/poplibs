// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_GraphUtil_hpp
#define poplibs_support_GraphUtil_hpp

#include <boost/graph/adjacency_matrix.hpp>
#include <vector>

// Returns an adjacency matrix representing pair-wise reachability of the input
// graph.
//
// Complexity is O(v^3) where v is the number of vertices in the given graph.
template <typename Graph>
static inline boost::adjacency_matrix<> pairwise_reachability(const Graph &g) {

  // Use a variant of the Floyd-Warshall algorithm to compute pairwise
  // reachability of the vertices in the graph.
  const auto numV = boost::num_vertices(g);
  std::vector<bool> r(numV * numV);

  // NOTE: Self-reachability is irrelevant to the algorithm.
  // The resulting graph defining reachability has a vertex v
  // adjacent to itself, i.e. reachable from itself.
  const auto &edges = boost::edges(g);
  for (auto it = edges.first; it != edges.second; ++it) {
    const auto u = boost::source(*it, g);
    const auto v = boost::target(*it, g);
    r[u * numV + v] = true;
  }

  // O(v^3) complexity comes from here.
  for (std::size_t k = 0; k < numV; ++k) {
    for (std::size_t u = 0; u < numV; ++u) {
      for (std::size_t v = 0; v < numV; ++v) {
        r[u * numV + v] =
            r[u * numV + v] || (r[u * numV + k] && r[k * numV + v]);
      }
    }
  }

  boost::adjacency_matrix<> result(numV);
  for (std::size_t u = 0; u < numV; ++u) {
    for (std::size_t v = 0; v < numV; ++v) {
      if (r[u * numV + v]) {
        boost::add_edge(u, v, result);
      }
    }
  }
  return result;
}

#endif // poplibs_support_GraphUtil_hpp
