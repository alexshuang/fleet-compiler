from __future__ import annotations

import numpy as np
from fleet_compiler.ir.core import Operation

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
        is_lhs_ranked_tensor_type = is_rhs_ranked_tensor_type = False
        if isinstance(lhs_type := op.operands[0].type, RankedTensorType):
            is_lhs_ranked_tensor_type = True
        if isinstance(rhs_type := op.operands[1].type, RankedTensorType):
            is_rhs_ranked_tensor_type = True
        if is_lhs_ranked_tensor_type == is_rhs_ranked_tensor_type == False:
            import pdb; pdb.set_trace()
            raise ValueError("All operands are unranked tensors!")

        if is_lhs_ranked_tensor_type and not is_rhs_ranked_tensor_type:
            op.operands[1].type = lhs_type
        elif not is_lhs_ranked_tensor_type and is_rhs_ranked_tensor_type:
            op.operands[0].type = rhs_type

        op.results[0].type = op.operands[0].type


class _UnaryOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, input: Value):
        super().__init__(operands=[input], result_types=[input.type])

    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
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
    hasCanonicalizer = True

    def __init__(self, input: Value, attrs: dict[str, Attribute]):
        attr = attrs['new_shape']
        type = RankedTensorType(attr.value, input.type.element_type)
        super().__init__(operands=[input], result_types=[type], attributes=attrs)
    
    def get_canonicalize_patterns(self):
        from ..transforms.canonicalize_patterns.tosa import (
            RemoveRedundantReshape
        )
        return [RemoveRedundantReshape()]


class MatmulOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, lhs: Value, rhs: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(lhs.type, RankedTensorType) and isinstance(rhs.type, RankedTensorType):
            lhs_dims, rhs_dims = lhs.type.dims, rhs.type.dims
            assert len(lhs_dims) == len(rhs_dims) == 2
            output_dims = [lhs_dims[0], rhs_dims[1]]
            output_type = RankedTensorType(output_dims, lhs.type.element_type)
        else:
            output_type = UnrankedTensorType(lhs.type.element_type)
        super().__init__(operands=[lhs, rhs], result_types=[output_type], attributes=attrs)
    
    def infer_shapes(self, op: Operation):
        lhs_shape = op.operands[0].type.dims
        rhs_shape = op.operands[1].type.dims

        if len(lhs_shape) == len(rhs_shape) and len(rhs_shape) >= 3:
            assert lhs_shape[:2] == rhs_shape[:2]

        lhs_k = lhs_shape[-1]
        rhs_k, n = rhs_shape[-2:]
        assert lhs_k == rhs_k
        output_shape = lhs_shape[:-1] + [n]
        op.results[0].type = RankedTensorType(output_shape,
                                              op.operands[0].type.element_type)


class GatherOp(Operation, ShapeInferenceOpInterface):
    hasCanonicalizer = True

    def __init__(self, values: Value, indices: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(values.type, RankedTensorType) and isinstance(indices.type, RankedTensorType):
            values_dims, indices_dims = values.type.dims, indices.type.dims
            output_dims = indices_dims + [values_dims[-1]]
            output_type = RankedTensorType(output_dims, values.type.element_type)
        else:
            output_type = UnrankedTensorType(values.type.element_type)
        super().__init__(operands=[values, indices], result_types=[output_type], attributes=attrs)

    def infer_shapes(self, op: Operation):
        dims = op.operands[0].type.dims.copy()
        id_dims = op.operands[1].type.dims
        for i, o in enumerate(id_dims):
            dims[i] = o

        # squeeze
        if len(dims) > 1:
            dims = [o for o in dims if o != 1]

        op.results[0].type = RankedTensorType(dims, op.operands[0].type.element_type)

    def get_canonicalize_patterns(self):
        from ..transforms.canonicalize_patterns.tosa import (
            RemoveCastedIndiceOperandForGather,
            CastIndiceIntergeToTensor,
        )
        return [RemoveCastedIndiceOperandForGather(), CastIndiceIntergeToTensor()]


class PowOp(Operation, ShapeInferenceOpInterface):
    def __init__(self, input1: Value, input2: Value, attrs: dict[str, Attribute] = {}):
        if isinstance(input1.type, RankedTensorType):
            output_type = input1.type
        else:
            output_type = UnrankedTensorType(input1.type.element_type)
        super().__init__(operands=[input1, input2], result_types=[output_type], attributes=attrs)
    
    def infer_shapes(self, op: Operation):
        if isinstance(in_type := op.operands[0].type, UnrankedTensorType):
            raise ValueError(f"operand is unranked tensors")
        op.results[0].type = in_type


class ReduceSumOp(Operation):
    name = "reduce_sum"

    def __init__(self, input: Value, attrs: dict[str, Attribute]):
        try:
            axis = attrs['axis'].value
            dims = input.type.dims.copy()
            dims[axis] = 1
            output_type = RankedTensorType(dims, input.type.element_type)
        except Exception as e:
            print(f"ERROR: {e}")
            raise
        super().__init__(operands=[input], result_types=[output_type], attributes=attrs)


class ConstOp(Operation):
    def __init__(self, attrs: dict[str, Attribute]):
        assert 'value' in attrs, f"value attribute not found"
        super().__init__(result_types=[attrs['value'].type], attributes=attrs)


class CastOp(Operation):
    hasCanonicalizer = True

    def __init__(self, input: Value, result_type: IRType):
        super().__init__(operands=[input], result_types=[result_type])

    def get_canonicalize_patterns(self):
        from ..transforms.canonicalize_patterns.tosa import (
            RemoveRedundantCast
        )
        return [RemoveRedundantCast()]
