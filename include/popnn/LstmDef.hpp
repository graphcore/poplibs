#ifndef __popnn_LstmDef_hpp__
#define __popnn_LstmDef_hpp__


/**
 * The units within a basic LSTM cell. The term unit is used to refer to either
 * a gate, or a cell state vector computation. In general all of these
 * require a weight matrix, a bias and a non-linearity. Typically, a fixed
 * type of non-linearity is associated with each type of unit.
 */
enum BasicLstmCellUnit {BASIC_LSTM_CELL_FORGET_GATE,
                        BASIC_LSTM_CELL_INPUT_GATE,
                        BASIC_LSTM_CELL_CANDIDATE,
                        BASIC_LSTM_CELL_OUTPUT_GATE,
                        BASIC_LSTM_CELL_NUM_UNITS
                       };

#endif // _popnn_LstmDef_hpp__
