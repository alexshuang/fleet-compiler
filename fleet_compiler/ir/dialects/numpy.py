from __future__ import annotations

from ..core import *
from .builtin import *


class Random_RandnOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        output_shape = [o.owner().attributes['value'].value for o in args]
        output_type = RankedTensorType(output_shape, FloatType(32))
        super().__init__(operands=args, result_types=[output_type])


class TransposeOp(Operation):
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
        super().__init__(operands=operands, result_types=[output_type])


class MeanOp(Operation):
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


class VarOp(Operation):
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


class SqrtOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])


class Random_SeedOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        output_type = NoneType()
        super().__init__(operands=args, result_types=[output_type])


class MaxOp(Operation):
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


class SumOp(Operation):
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


class ExpOp(Operation):
    def __init__(self, args: list[Value], kwargs: dict[str, Value]):
        super().__init__(operands=args, result_types=[args[0].type])
