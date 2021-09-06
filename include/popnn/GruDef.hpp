// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *  Definitions for GRU cell operations.
 */

#ifndef popnn_GruDef_hpp
#define popnn_GruDef_hpp

/**
 * The units within a basic GRU cell. In general all of these
 * require a weight matrix, a bias and a non-linearity. Typically,
 * a fixed type of non-linearity is associated with each type of unit.
 */
enum BasicGruCellUnit {
  BASIC_GRU_CELL_RESET_GATE = 0,
  BASIC_GRU_CELL_UPDATE_GATE = 1,
  BASIC_GRU_CELL_CANDIDATE = 2,
  BASIC_GRU_CELL_NUM_UNITS = 3
};

#endif // popnn_GruDef_hpp
