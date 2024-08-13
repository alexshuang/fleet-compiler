from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from ..core import *


## Types
@dataclass
class StringType(IRType): ...


@dataclass
class IntegerType(IRType):
    bitwidth: int
    signedness: bool


@dataclass
class FloatType(IRType):
    bitwidth: int


@dataclass
class BoolType(IRType): ...


@dataclass
class NoneType(IRType): ...


@dataclass
class RankedTensorType(IRType):
    dims: list
    element_type: IntegerType | FloatType


@dataclass
class UnrankedTensorType(IRType):
    element_type: IntegerType | FloatType


@dataclass
class ArrayType(IRType):
    dims: int
    element_type: IntegerType | FloatType


## Attributes
@dataclass
class StringAttr(Attribute):
    value: str
    type: StringType = field(init=False, default_factory=StringType)


@dataclass
class IntegerAttr(Attribute):
    value: int
    type: IntegerType


@dataclass
class FloatAttr(Attribute):
    value: float
    type: FloatType


@dataclass
class BoolAttr(Attribute):
    value: bool


@dataclass
class NoneAttr(Attribute): ...


@dataclass
class DenseIntOrFPElementsAttr(Attribute):
    value: np.ndarray | list[int | float] | int | float
    type: RankedTensorType | UnrankedTensorType


@dataclass
class ArrayAttr(Attribute):
    value: list
    type: ArrayType


def get_preference_by_type_cls(type: IRType):
    if isinstance(type, IntegerType):
        return 1
    elif isinstance(type, FloatType):
        return 2
    elif isinstance(type, RankedTensorType):
        return 3
    elif isinstance(type, UnrankedTensorType):
        return 4
    else:
        raise ValueError(f"unsupported type: {type}")


class ModuleOp(Operation):
    sym_name = StringAttr("main")
    body: Region

    def __init__(self, sym_name: str = "main", attributes: dict[str, Attribute] = None):
        self.sym_name = StringAttr(sym_name)
        self.body = Region([Block()])
        self.attributes = attributes if attributes else {}
        super().__init__(regions=[self.body],
                         attributes=self.attributes)

    @property
    def operations(self):
        return self.body.blocks[0].operations

    def dump(self, fp = None):
        from ..printer import Printer
        Printer(fp).print(self)
