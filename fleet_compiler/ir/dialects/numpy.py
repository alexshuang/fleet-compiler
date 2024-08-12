from __future__ import annotations

from fleet_compiler.ir.core import Operation

from ..core import *
from .builtin import *
from ..interfaces import ShapeInferenceOpInterface
from ..traits import *


class Random_RandnOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        output_shape = [o.owner().attributes['value'].value for o in args]
        output_type = RankedTensorType(output_shape, FloatType(32))
        super().__init__(operands=args, result_types=[output_type])


class TransposeOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        operands = [args[0]]

        axes = None
        if len(args) > 1:
            axes = args[1]
        elif len(kwargs) > 0:
            axes = kwargs['axes']

        input_type = args[0].type

        axes_value = None
        if axes:
            operands.append(axes)
            axes_value = axes.owner().attributes['value'].value
        elif isinstance(input_type, RankedTensorType):
            input_dims = input_type.dims
            axes_value = list(range(len(input_dims) - 1, -1, -1))

        if axes_value:
            attrs['axes'] = ArrayAttr(axes_value,
                                    ArrayType(len(axes_value), IntegerType(32, True)))
        else:
            attrs['axes'] = NoneAttr()

        if isinstance(input_type, RankedTensorType):
            output_dims = [input_dims[i] for i in axes_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)
    
    def infer_shapes(self, op: Operation):
        t = op.operands[0].type
        dims = t.dims
        axes_val = None if isinstance(axes := op.attributes['axes'], NoneAttr) else axes.value
        if axes_val:
            assert len(axes_val) == len(dims)
            new_dims = [dims[i] for i in axes_val]
        else:
            new_dims = dims[::-1]
        op.results[0].type = RankedTensorType(new_dims, t.element_type)


class MeanOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        operands = [args[0]]
        input_type = args[0].type
        input_dims = input_type.dims if isinstance(input_type, RankedTensorType) else None

        if len(args) >= 2:
            axis = args[1]
        elif len(kwargs) > 0:
            axis = kwargs['axis']

        if axis:
            operands.append(axis)
            axis_value = axis.owner().attributes['value'].value
            if not isinstance(axis_value, list):
                axis_value = [axis_value]
            attrs['axis'] = ArrayAttr(axis_value,
                                      ArrayType(len(axis_value), IntegerType(32, True)))

        if len(args) >= 5:
            keepdims = args[4]
        elif len(kwargs) > 0:
            keepdims = kwargs['keepdims']

        if keepdims:
            operands.append(keepdims)
            keepdims_value = keepdims.owner().attributes['value'].value
            attrs['keepdims'] = BoolAttr(keepdims_value)

        if input_dims:
            output_dims = input_dims[axis_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        axis = op.attributes['axis'].value[0]
        new_dims = [o if i != axis else 1 for i, o in enumerate(op.operands[0].type.dims)]
        op.results[0].type = RankedTensorType(new_dims, op.operands[0].type.element_type)


class VarOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        operands = [args[0]]
        input_type = args[0].type
        input_dims = input_type.dims if isinstance(input_type, RankedTensorType) else None

        if len(args) >= 2:
            axis = args[1]
        elif len(kwargs) > 0:
            axis = kwargs['axis']

        if axis:
            operands.append(axis)
            axis_value = axis.owner().attributes['value'].value
            if not isinstance(axis_value, list):
                axis_value = [axis_value]
            attrs['axis'] = ArrayAttr(axis_value,
                                      ArrayType(len(axis_value), IntegerType(32, True)))

        if len(args) >= 6:
            keepdims = args[5]
        elif len(kwargs) > 0:
            keepdims = kwargs['keepdims']

        if keepdims:
            operands.append(keepdims)
            keepdims_value = keepdims.owner().attributes['value'].value
            attrs['keepdims'] = BoolAttr(keepdims_value)

        if input_dims:
            output_dims = input_dims[axis_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        axis = op.attributes['axis'].value
        new_dims = [o if i != axis else 1 for i, o in enumerate(op.operands[0].type.dims)]
        op.results[0].type = RankedTensorType(new_dims, op.operands[0].type.element_type)


class SqrtOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])

    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
        op.results[0].type = in_type


class Random_SeedOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        output_type = NoneType()
        super().__init__(operands=args, result_types=[output_type], traits=[Pure()])


class MaxOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        operands = [args[0]]
        input_type = args[0].type
        input_dims = input_type.dims if isinstance(input_type, RankedTensorType) else None

        if len(args) >= 2:
            axis = args[1]
        elif len(kwargs) > 0:
            axis = kwargs['axis']

        if axis:
            operands.append(axis)
            axis_value = axis.owner().attributes['value'].value
            if not isinstance(axis_value, list):
                axis_value = [axis_value]
            attrs['axis'] = ArrayAttr(axis_value,
                                      ArrayType(len(axis_value), IntegerType(32, True)))

        if len(args) >= 5:
            keepdims = args[4]
        elif len(kwargs) > 0:
            keepdims = kwargs['keepdims']

        if keepdims:
            operands.append(keepdims)
            keepdims_value = keepdims.owner().attributes['value'].value
            attrs['keepdims'] = BoolAttr(keepdims_value)

        if input_dims:
            output_dims = input_dims[axis_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        axis = op.attributes['axis'].value[0]
        new_dims = [o if i != axis else 1 for i, o in enumerate(op.operands[0].type.dims)]
        op.results[0].type = RankedTensorType(new_dims, op.operands[0].type.element_type)


class SumOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        operands = [args[0]]
        input_type = args[0].type
        input_dims = input_type.dims if isinstance(input_type, RankedTensorType) else None

        if len(args) >= 2:
            axis = args[1]
        elif len(kwargs) > 0:
            axis = kwargs['axis']

        if axis:
            operands.append(axis)
            axis_value = axis.owner().attributes['value'].value
            if not isinstance(axis_value, list):
                axis_value = [axis_value]
            attrs['axis'] = ArrayAttr(axis_value,
                                      ArrayType(len(axis_value), IntegerType(32, True)))

        if len(args) >= 5:
            keepdims = args[4]
        elif len(kwargs) > 0:
            keepdims = kwargs['keepdims']

        if keepdims:
            operands.append(keepdims)
            keepdims_value = keepdims.owner().attributes['value'].value
            attrs['keepdims'] = BoolAttr(keepdims_value)

        if input_dims:
            output_dims = input_dims[axis_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        axis = op.attributes['axis'].value[0]
        new_dims = [o if i != axis else 1 for i, o in enumerate(op.operands[0].type.dims)]
        op.results[0].type = RankedTensorType(new_dims, op.operands[0].type.element_type)


class ExpOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])
    
    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
        op.results[0].type = in_type


class SplitOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        attrs = {}
        assert len(args) >= 2, "Invalid arguments"
        operands = [args[0], args[1]]
        input_type = args[0].type
        input_dims = input_type.dims if isinstance(input_type, RankedTensorType) else None

        axis = None
        if len(args) >= 3:
            axis = args[2]
        elif len(kwargs) > 0:
            axis = kwargs['axis']

        if axis:
            operands.append(axis)
            axis_value = axis.owner().attributes['value'].value
            if not isinstance(axis_value, list):
                axis_value = [axis_value]
            attrs['axis'] = ArrayAttr(axis_value,
                                      ArrayType(len(axis_value), IntegerType(32, True)))

        if input_dims:
            output_dims = input_dims[axis_value]
            output_type = RankedTensorType(output_dims, input_type.element_type)
        else:
            output_type = UnrankedTensorType(input_type.element_type)
        super().__init__(operands=operands, result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        dims = op.operands[0].type.dims.copy()
        tile_size = op.operands[1].owner().attributes['value'].value
        axis = op.attributes['axis'].value[0]
        assert dims[axis] % tile_size == 0
        dims[axis] = dims[axis] // tile_size
        dims = [tile_size] + dims
        op.results[0].type = RankedTensorType(dims, op.operands[0].type.element_type)


class TriOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])
    
    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
        op.results[0].type = in_type


class HstackOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        assert len(args) == 1 and len(args[0]) >= 2
        args = args[0]
        super().__init__(operands=args, result_types=[args[0].type])

    def infer_shapes(self, op: Operation):
        batches = op.operands[0].type.dims[:-1]
        elem_type = op.operands[0].type.element_type
        dim = 0
        for o in op.operands:
            assert o.type.dims[:-1] == batches
            assert o.type.element_type == elem_type
            dim += o.type.dims[-1]
        op.results[0].type = RankedTensorType(batches + [dim], elem_type)


class TanhOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])
    
    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
        op.results[0].type = in_type
