#include <popnn/SpatialSoftMax.hpp>
#include <popops/ElementWise.hpp>
#include <poplin/MeshGrid.hpp>
#include <poplin/MatMul.hpp>
#include <poplar/exceptions.hpp>
#include <poplar/VariableMappingMethod.hpp>

namespace popnn {

std::pair<poplar::Tensor, poplar::Tensor>
spatialSoftMax2D(poplar::Graph& graph, poplar::program::Sequence& prog,
                 const poplar::Tensor &fields, float initialTemperature,
                 bool disableSoftmax, const std::string &name) {
  if (fields.rank() != 3) {
    throw poplar::poplar_error("In spatialSoftMax2D "
                               "fields tensor must have rank 3");
  }

  // Add new variables:
  const poplar::Type type = fields.elementType();
  auto temperature = graph.addVariable(type, {}, name + "/temperature");
  graph.setInitialValue(temperature, initialTemperature);
  graph.setTileMapping(temperature, 0);
  auto one = graph.addConstant(type, {}, 1.f);

  // Do a scalar divide and then multiply by the scale factor:
  auto scale = popops::div(graph, one, temperature, prog,
                           name + "/scale_factor");
  auto fieldsScaled = popops::mul(graph, fields, scale, prog,
                                  name + "/fields_scaled");

  // Perform softmax (if enabled) over all inputs jointly (flattened):
  auto fieldsSoftMaxFlat = fieldsScaled.flatten();
  if (!disableSoftmax) {
    nonLinearity(graph, popnn::NonLinearityType::SOFTMAX, fieldsSoftMaxFlat,
                 prog, name + "/softmax");
  }

  // Add variables for the axes coordinates and grid them:
  const auto width  = fields.dim(2);
  const auto height = fields.dim(1);
  auto xCoords = poplin::linspace(graph, type, -1.f, 1.f, width);
  auto yCoords = poplin::linspace(graph, type, -1.f, 1.f, height);
  auto grids = poplin::meshgrid2d(graph, xCoords, yCoords);
  poplar::Tensor &xGrid = grids.at(0);
  poplar::Tensor &yGrid = grids.at(1);

  // To compute the SSM: pointwise multiply each field with both the x and y
  // grids (separately) then take the sum of all the resulting elements which
  // gives an expected x and y coordinate for each field.
  // Do this efficiently by matricising the scaled fields with a reshape, and
  // then matricise the grids by flattening and concat, then the pointwise
  // multiplies and reductions become a single matrix multiply with shape
  // [F, WxH] * [WxH, 2] -> [F, 2].
  // The expected x coords for each input field run down the second column of
  // the result matrix and the expected y's run down the first column:

  // LHS matrix has one entire softmax field flattened along each row
  // (so cols contain width*height elements):
  const auto fieldSize = width*height;
  const auto numFields = fields.dim(0);
  const auto lhsShape = std::vector<std::size_t>{numFields, fieldSize};
  const auto rhsShape = std::vector<std::size_t>{fieldSize, 2u};
  auto lhs = fieldsSoftMaxFlat.reshape(lhsShape);

  // RHS matrix has ys flattened into the first column and xs
  // flattened into the second column. The reversal is in order to preserve
  // a consistent (row, col) indexing order in the result (Poplar is row major).
  std::vector<poplar::Tensor> colTensors;
  colTensors.push_back(yGrid.reshape({fieldSize, 1}));
  colTensors.push_back(xGrid.reshape({fieldSize, 1}));
  auto rhs = poplar::concat(colTensors, 1).reshape(rhsShape);

  // Create tensors with optimal layout for the shape of the matrix multiply
  // and then copy the lhs data and set the tile mapping for the rhs (as it is
  // a view on a constant):
  auto l = poplin::createMatMulInputLHS(graph, type, lhsShape, rhsShape,
                                        name + "/lhs");
  auto r = poplin::createMatMulInputRHS(graph, type, lhsShape, rhsShape,
                                        name +"/rhs");
  prog.add(poplar::program::Copy(lhs, l));
  prog.add(poplar::program::Copy(rhs, r));

  // Return the matrix multiply result variable and the temperature variable:
  return {poplin::matMul(graph, l, r, prog, name + "/mat_mul"), temperature};
}

} // end namespace popnn
