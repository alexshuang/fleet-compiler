
# ===- builder.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

from .core import *
from dataclasses import dataclass, field
import contextlib


@dataclass(frozen=True)
class InsertionPoint:
    block: Block
    insert_before: Operation = field(default=None)

    def __post_init__(self):
        if self.insert_before is not None:
            if self.insert_before.parent is not self.block:
                raise ValueError("The insertion position is not in the target block")
    
    @staticmethod
    def before(op: Operation):
        return InsertionPoint(op.parent, op)

    @staticmethod
    def at_end(block: Block):
        return InsertionPoint(block)


@dataclass
class BuilderStack:
    stack: Sequence[Builder] = field(default_factory=list)

    def push(self, b: Builder):
        self.stack.append(b)
    
    def pop(self):
        return self.stack.pop()
    
    def get(self):
        return self.stack[-1]
    
    def walk(self):
        return self.stack[::-1]


@dataclass
class SymbolTable:
    "A mapping from variable names to Values, append-only"
    table: dict[str, Value | str] = field(default_factory=dict)

    def __contains__(self, __o: object) -> bool:
        return __o in self.table

    def __getitem__(self, __key: str) -> Value:
        return self.table[__key]

    def __setitem__(self, __key: str, __value: Value) -> None:
        self.table[__key] = __value


class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImplicitBuilder(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._stack = BuilderStack()

    def push(self, builder: Builder):
        self._stack.push(builder)
        
    def pop(self):
        return self._stack.pop()
    
    def get(self):
        return self._stack.get()
    
    def lookup_symbol(self, sym_name: str):
        def get_value_by_symbol(builder: Builder, sym_name: str):
            val = sym_name
            # support str mapping to str, like {'a': 'arg0', 'arg0': Value}
            while isinstance(val, str) and val in builder.symbol_table:
                val = builder.symbol_table[val]
            return val if isinstance(val, Value) else None

        for i, b in enumerate(self._stack.walk()):
            if val := get_value_by_symbol(b, sym_name):
                return (val, i)
        raise ValueError(f"unkown symbol {sym_name} in symbol table stack")


class Builder(contextlib.AbstractContextManager):
    def __init__(self, insertion_point: InsertionPoint) -> None:
        super().__init__()
        self.insertion_point = insertion_point
        self.symbol_table = SymbolTable()

    @staticmethod
    def before(op: Operation):
        return Builder(InsertionPoint.before(op))

    @staticmethod
    def at_end(block: Block):
        return Builder(InsertionPoint.at_end(block))
    
    def insert(self, op: Operation):
        ip = self.insertion_point
        if ip.insert_before:
            ip.block.insert_before(op, ip.insert_before)
        else:
            ip.block.add_op(op)

    def __enter__(self) -> Any:
        ImplicitBuilder().push(self)
        return ImplicitBuilder().get()

    def __exit__(self, __exc_type: type[BaseException] | None,
                 __exc_value: BaseException | None,
                 __traceback: contextlib.TracebackType | None) -> bool | None:
        return ImplicitBuilder().pop()
