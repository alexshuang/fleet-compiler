from __future__ import annotations

from .func import CallOp, FuncOp, ReturnOp
from .tosa import ConstOp
from .builtin import ModuleOp


class Const_F32Op(ConstOp): ...

class Const_F32_ZeroOp(ConstOp): ...

class Const_I32Op(ConstOp): ...

class Const_I32_ZeroOp(ConstOp): ...

class RodataOp(ConstOp): ...


class CallOp(CallOp):
    hasCanonicalizer = True

    def get_canonicalize_patterns(self):
        from ..transforms.canonicalize_patterns.vm import (
            ReplaceReciprocalMulWithDiv
        )
        return [ReplaceReciprocalMulWithDiv()]


class FuncOp(FuncOp): ...

class ReturnOp(ReturnOp): ...

class ModuleOp(ModuleOp): ...
