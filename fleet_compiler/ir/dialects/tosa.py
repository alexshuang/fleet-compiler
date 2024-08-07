from __future__ import annotations

from ..core import *
from .builtin import *
from ..interfaces import ShapeInferenceOpInterface


class _BinaryOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        output_type = lhs.type
        if lhs.type != rhs.type:
            lhs_perf = get_preference_by_type_cls(lhs.type)
            rhs_perf = get_preference_by_type_cls(rhs.type)
            output_type = lhs.type if lhs_perf > rhs_perf else rhs.type
        super().__init__(operands=[lhs, rhs], result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        if isinstance(lhs_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"Invalid operand type: {lhs_type}")
        if isinstance(rhs_type := op.operands[1].type, UnrankedTensorType):
            raise ValueError(f"Invalid operand type: {rhs_type}")
        op.results[0].type = lhs_type


class _UnaryOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, input: Value):
        super().__init__(operands=[input], result_types=[input.type])

    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"Invalid operand type: {in_type}")
        op.results[0].type = in_type


class AddOp(_BinaryOp): ...


class SubOp(_BinaryOp): ...


class MulOp(_BinaryOp):
    def __init__(self, lhs: Value, rhs: Value):
        # not support quantlization
        attrs = {'shift': IntegerAttr(0, IntegerType(32, True))}
        super().__init__(lhs, rhs, attrs)


class ReciprocalOp(_UnaryOp): ...


class ReshapeOp(Operation):
    def __init__(self, input: Value, attrs: dict[str, Attribute]):
        attr = attrs['new_shape']
        type = RankedTensorType(attr.value, input.type.element_type)
        super().__init__(operands=[input], result_types=[type], attributes=attrs)


class MatmulOp(Operation):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(lhs.type, RankedTensorType) and isinstance(rhs.type, RankedTensorType):
            lhs_dims, rhs_dims = lhs.type.dims, rhs.type.dims
            assert len(lhs_dims) == len(rhs_dims) == 2
            output_dims = [lhs_dims[0], rhs_dims[1]]
            output_type = RankedTensorType(output_dims, lhs.type.element_type)
        else:
            output_type = UnrankedTensorType(lhs.type.element_type)
        super().__init__(operands=[lhs, rhs], result_types=[output_type], attributes=attrs)


class GatherOp(Operation):
    def __init__(self, values: Value, indices: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(values.type, RankedTensorType) and isinstance(indices.type, RankedTensorType):
            values_dims, indices_dims = values.type.dims, indices.type.dims
            output_dims = indices_dims + [values_dims[-1]]
            output_type = RankedTensorType(output_dims, values.type.element_type)
        else:
            output_type = UnrankedTensorType(values.type.element_type)
        super().__init__(operands=[values, indices], result_types=[output_type], attributes=attrs)


class PowOp(Operation):
    def __init__(self, input1: Value, input2: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(input1.type, RankedTensorType):
            output_type = input1.type
        else:
            output_type = UnrankedTensorType(input1.type.element_type)
        super().__init__(operands=[input1, input2], result_types=[output_type], attributes=attrs)
