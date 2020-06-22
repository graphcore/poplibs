// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "ZoltanPartitioner.hpp"
#include <memory>
#include <poplibs_support/logging.hpp>
#include <poputil/exceptions.hpp>
#include <zoltan_cpp.h>

namespace logging = poplibs_support::logging;

namespace popsparse {
namespace experimental {

static int objectNumber(void *data, int *ierr) {
  // Returns the number of vertices in the hypergraph.

  auto &graph = *static_cast<HyperGraphData *>(data);
  *ierr = ZOLTAN_OK;
  return static_cast<int>(graph.nodes);
}

static void objectList(void *data, int globalIDEntries, int localIDEntries,
                       ZOLTAN_ID_PTR globalIDs, ZOLTAN_ID_PTR localIDs,
                       int weightDimension, float *objectWeights, int *ierr) {
  // Fills in the global IDs and weights for the vertices in the hypergraph.

  auto &graph = *static_cast<HyperGraphData *>(data);
  *ierr = ZOLTAN_OK;

  for (unsigned i = 0; i < graph.nodes; ++i) {
    globalIDs[i] = i;
  }

  if (weightDimension > 0) {
    for (unsigned i = 0; i < graph.weights.size(); ++i) {
      objectWeights[i] = graph.weights[i];
    }
  }
}

static void hypergraphSize(void *data, int *hyperGraphSize, int *pinCount,
                           int *format, int *ierr) {
  // Fills in the number of hyperedges and the number of vertices in those
  // edges.

  auto &graph = *static_cast<HyperGraphData *>(data);
  *ierr = ZOLTAN_OK;

  *hyperGraphSize = static_cast<int>(graph.hyperEdges.size());
  *pinCount = static_cast<int>(graph.pins.size());
  *format = ZOLTAN_COMPRESSED_EDGE;
}

static void hypergraphList(void *data, int globalIDEntries, int hyperGraphSize,
                           int pinCount, int format, ZOLTAN_ID_PTR hyperEdgeGID,
                           int *hyperEdgeOffset, ZOLTAN_ID_PTR pinGID,
                           int *ierr) {
  // Fills in the the global IDs of the hyperedges and the global IDs of the
  // vertices in each edge.

  auto &graph = *static_cast<HyperGraphData *>(data);
  *ierr = ZOLTAN_OK;

  for (int i = 0; i < hyperGraphSize; ++i) {
    hyperEdgeGID[i] = i;
    hyperEdgeOffset[i] = static_cast<int>(graph.hyperEdges[i]);
  }

  for (int i = 0; i < pinCount; ++i) {
    pinGID[i] = graph.pins[i];
  }
}

bool ZoltanPartitioner::partitionGraph(const HyperGraphData &graphData,
                                       int nPartition,
                                       std::vector<int> &nodeAssignment) {
  float zoltanVersion;

  if (Zoltan_Initialize(0, nullptr, &zoltanVersion) != ZOLTAN_OK) {
    throw poputil::poplibs_error("Partitioning of the graph failed");
    return false;
  }

  std::unique_ptr<Zoltan> zz(new Zoltan);

  logging::info("Zoltan version: {}", zoltanVersion);

  // Register query functions.
  zz->Set_Num_Obj_Fn(objectNumber, (void *)&graphData);
  zz->Set_Obj_List_Fn(objectList, (void *)&graphData);
  zz->Set_HG_Size_CS_Fn(hypergraphSize, (void *)&graphData);
  zz->Set_HG_CS_Fn(hypergraphList, (void *)&graphData);

  // Set parameters
  // Set debug level to same level as POPLIBS_LOG_LEVEL.
  zz->Set_Param("DEBUG_LEVEL", "0");

  // We want to use hypergraphs to model our connections.
  if (partitionType == PartitionType::BLOCK) {
    zz->Set_Param("LB_METHOD", "BLOCK");
  } else {
    zz->Set_Param("LB_METHOD", "HYPERGRAPH");
  }

  // PHG is the Zoltan hypergraph partitioning package.
  zz->Set_Param("HYPERGRAPH_PACKAGE", "PHG");

  // We have one and only one list of global ID entries.
  zz->Set_Param("NUM_GID_ENTRIES", "1");

  // We do not have any local ID entries.
  zz->Set_Param("NUM_LID_ENTRIES", "0");

  // We want to partition the hypergraph.
  zz->Set_Param("LB_APPROACH", "PARTITION");

  // We want to partition it into a predermined number of subgraphs.
  zz->Set_Param("NUM_GLOBAL_PARTS", std::to_string(nPartition));

  // Return the parts into which the vertices are partitioned.
  zz->Set_Param("RETURN_LISTS", "PARTS");

  // We have a one-dimensional list of object weights.
  zz->Set_Param("OBJ_WEIGHT_DIM", "1");

  // We do not have any edge weights.
  zz->Set_Param("EDGE_WEIGHT_DIM", "0");

  // TODO:
  // Would it be beneficial to set edge weights,
  // in particular if the partial type is larger than the input type
  // then edge that connects the output to the calculation should have a larger
  // weight than the edges that connect the inputs to the calculation?

  // Perform the partitioning.

  struct PartitionData {
    int changes = 0;
    int globalIDEntries = 1;
    int localIDEntries = 0;
    int imports = 1;
    ZOLTAN_ID_PTR importGlobalIDs = nullptr;
    ZOLTAN_ID_PTR importLocalIDs = nullptr;
    int *importProcs = nullptr;
    int *importToPart = nullptr;
    int exports = 1;
    ZOLTAN_ID_PTR exportGlobalIDs = nullptr;
    ZOLTAN_ID_PTR exportLocalIDs = nullptr;
    int *exportProcs = nullptr;
    int *exportToPart = nullptr;

    ~PartitionData() {
      Zoltan::LB_Free_Part(&importGlobalIDs, &importLocalIDs, &importProcs,
                           &importToPart);
      Zoltan::LB_Free_Part(&exportGlobalIDs, &exportLocalIDs, &exportProcs,
                           &exportToPart);
    }
  } data;

  auto result = zz->LB_Partition(
      data.changes, data.globalIDEntries, data.localIDEntries, data.imports,
      data.importGlobalIDs, data.importLocalIDs, data.importProcs,
      data.importToPart, data.exports, data.exportGlobalIDs,
      data.exportLocalIDs, data.exportProcs, data.exportToPart);

  switch (result) {
  case ZOLTAN_OK:
    break;
  case ZOLTAN_WARN:
    logging::warn("Hypergraph partitioning returned with warnings");
    break;
  case ZOLTAN_FATAL:
    throw poputil::poplibs_error("Partitioning of the hypergraph failed");
  case ZOLTAN_MEMERR:
    throw poputil::poplibs_error(
        "Memory allocation failure in hypergraph partitioning");
  }

  // Translate the partition back into a tile mapping.
  nodeAssignment.resize(data.exports);

  for (int i = 0; i < data.exports; ++i) {
    nodeAssignment[data.exportGlobalIDs[i]] = data.exportToPart[i];
  }

  return true;
}

} // namespace experimental
} // namespace popsparse