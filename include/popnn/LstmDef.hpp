// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *  Definitions for LSTM cell operations.
 */

#ifndef popnn_LstmDef_hpp
#define popnn_LstmDef_hpp

/**
 * The units within a basic LSTM cell.
 *
 * The term unit is used to refer to either
 * a gate, or a cell state vector computation. In general all of these
 * require a weight matrix, a bias and a non-linearity. Typically, a fixed
 * type of non-linearity is associated with each type of unit.
 */
enum BasicLstmCellUnit {
  BASIC_LSTM_CELL_FORGET_GATE = 0,
  BASIC_LSTM_CELL_INPUT_GATE = 1,
  BASIC_LSTM_CELL_CANDIDATE = 2,
  BASIC_LSTM_CELL_OUTPUT_GATE = 3,
  BASIC_LSTM_CELL_NUM_UNITS = 4
};

#endif // popnn_LstmDef_hpp
