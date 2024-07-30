# ===- core.py -------------------------------------------------------------
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

import re
from typing import Any, ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence, Iterable


op_result_id = 0


class IRNode(ABC):
    @property
    @abstractmethod
    def parent_node(self) -> IRNode | None: ...

    def is_ancestor(self, op: IRNode):
        if op is self:
            return True 
        if (parent := op.parent_node) is None:
            return False
        return self.is_ancestor(parent)


@dataclass
class Use:
    operation: Operation
    index: int
    

@dataclass
class IRType(ABC):
    name: str


@dataclass
class Value(ABC):
    _name: str
    type: IRType
    uses: Sequence[Use]
    _name_regex = re.compile(r"(%[A-Za-z_0-9]*)")

    def __post_init__(self):
        self.uses = []

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, data) -> str:
        if Value.is_valid_name(data):
            self._name = data
        else:
            raise ValueError(f"Invalid value name: {data}")
    
    @property
    @abstractmethod
    def owner(self) -> Operation | Block:
        pass
    
    @classmethod
    def is_valid_name(self, name: str | None):
        return True if name is None or re.fullmatch(self._name_regex) else False
    
    def replace_by(self, new_value: Value):
        for use in self.uses:
            use.operation.operands[use.index] = new_value


@dataclass
class Block(IRNode):
    name: str = ""
    arguments: Sequence[BlockArgument] = field(default_factory=list)
    operations: Sequence[Operation] = field(default_factory=list)
    parent: Region | None = None

    @property
    def parent_node(self):
        return self.parent
    
    def _attach_op(self, op: Operation):
        if op.parent:
            raise ValueError("Can't attach a attached op (already attached to a block)")
        if self.is_ancestor(op):
            raise ValueError("Can't attach a op which from children's block")
        op.parent = self
            
    def add_op(self, op: Operation):
        self.operations.append(op)
        self._attach_op(op)

    def add_ops(self, ops: list[Operation]):
        self.operations.extend(ops)
        for o in ops:
            self._attach_op(o)
    
    def insert_before(self, new_op: Operation, existing_op: Operation):
        assert existing_op.parent is self, "The \'existing op\' is not in current block"
        self._attach_op(new_op)
        idx = self.get_op_index(existing_op)
        self.operations.insert(idx, new_op)
    
    def get_op_index(self, op: Operation):
        assert op.parent is self, "The \'existing op\' is not in current block"
        return self.operations.index(op)
    
    def walk(self):
        for o in self.operations:
            yield from o.walk()
    
    def erase_op(self, op: Operation):
        old = self.operations.pop(self.get_op_index(op))
        old.parent = None

    def __hash__(self) -> int:
        return id(self)


@dataclass
class BlockArgument(Value):
    block: Block
    # index of the arguments
    index: int
    # AST arg name
    ast_name: str

    def owner(self) -> Block:
        return self.block

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Region(IRNode):
    blocks: Sequence[Block] = field(default_factory=list)
    parent: Operation | None = None

    # def __init__(self, blocks: Sequence[Block] = [], parent: Operation | None = None):
    def __post_init__(self):
        for b in self.blocks:
            self._attach_block(b)

    @property
    def parent_node(self):
        return self.parent

    @property
    def operations(self):
        if self.blocks and len(self.blocks) == 1:
            return self.blocks[0].operations
        else:
            raise ValueError(
                "'operations' property of Region class only available"
                " for single-block regions")

    def _attach_block(self, block: Block):
        if block.parent:
            raise ValueError("Can't attach a attached block (already attached to a region)")
        if self.is_ancestor(block):
            raise ValueError("Can't attach a block which from children's block")
        block.parent = self

    def add_block(self, block: Block):
        self._attach_block(block)
        block.name = f"^bb{len(self.blocks)}"
        self.blocks.append(block)

    def walk(self):
        for o in self.blocks:
            yield from o.walk()

    def clone(self, src_region: Region, block_mapper: dict[Block, Block] = {},
              value_mapper: dict[Value, Value] = {}):
        new_blocks = []
        for block in src_region.blocks:
            new_block = Block()
            block_mapper[block] = new_block
            new_blocks.append(new_block)

        for block, new_block in zip(src_region.blocks, new_blocks):
            for arg in block.arguments:
                new_arg = BlockArgument(arg.name, arg.type, [], new_block,
                                        arg.index, arg.ast_name)
                value_mapper[arg] = new_arg
                new_block.arguments.append(new_arg)
            for o in block.operations:
                new_block.add_op(o.clone())

        for o in new_blocks:
            self.add_block(o)


@dataclass
class OpResult(Value):
    # producer
    op: Operation
    # index of the results
    index: int

    def owner(self) -> Operation:
        return self.op

    def __hash__(self) -> int:
        return id(self)


@dataclass
class Attribute(ABC):
    pass


@dataclass
class IRType(ABC):
    pass


@dataclass
class Data(Attribute):
    value: Any


@dataclass
class OpTrait(ABC):
    pass


@dataclass
class Operation(IRNode):
    name = ""
    parent: Block | None = None
    operands: Sequence[OpResult | BlockArgument] = field(default_factory=list)
    results: Sequence[OpResult] = field(default_factory=list)
    successors: Sequence[Block] = field(default_factory=list)
    attributes: dict[str, Attribute] = field(default_factory=dict)
    properties: dict[str, Attribute] = field(default_factory=dict)
    regions: Sequence[Region] = field(default_factory=list)
    traits: Sequence[OpTrait] = field(default_factory=list)
    hasCustomAssemblyFormat: bool = field(default=False)
    
    op_result_id: ClassVar[int] = 0
    
    @property
    def parent_node(self) -> IRNode:
        return self.parent
    
    def __init__(self,
                 *,
                 operands: Sequence[OpResult | BlockArgument] = [],
                 result_types: Sequence[IRType] = [],
                 successors: Sequence[Block] = [],
                 attributes: dict[str, Attribute] = {},
                 properties: dict[str, Attribute] = {},
                 regions: Sequence[Region] = [],
                 traits: Sequence[OpTrait] = []):
        res = []
        for i, t in enumerate(result_types):
            res.append(OpResult(f'%{self.op_result_id}', t, uses=[], op=self, index=i))
            Operation.op_result_id += 1
        
        if self.name == "":
            if match := re.match(r'(.+)Op$', self.__class__.__name__):
                op_name = match.group(1).lower().replace('_', '.')
                self.name = f'{self.__class__.__module__.split(".")[-1]}.{op_name}'
        self.results = res
        self.operands = [o for o in operands]
        for i, o in enumerate(self.operands):
            o.uses.append(Use(self, i))
        self.successors = successors
        self.properties = properties
        self.attributes = attributes
        self.regions = []
        for o in regions:
            self.add_region(o)
        self.traits = traits
    
    @classmethod
    def create(cls,
            operands: Sequence[OpResult | BlockArgument] = [],
            result_types: Sequence[IRType] = [],
            successors: Sequence[Block] = [],
            attributes: dict[str, Attribute] = {},
            properties: dict[str, Attribute] = {},
            regions: Sequence[Region] = []):
        op = cls.__new__(cls)
        Operation.__init__(op,
                           operands=operands,
                           result_types=result_types,
                           successors=successors,
                           attributes=attributes,
                           properties=properties,
                           regions=regions)
        return op

    def add_region(self, region: Region):
        if region.parent:
            raise ValueError("Can't attach a attached region")
        region.parent = self
        self.regions.append(region)

    def walk(self):
        yield self
        for o in self.regions:
            yield from o.walk()

    def clone(self, block_mapper: dict[Block, Block] = {},
              value_mapper: dict[Value, Value] = {}):
        # import pdb; pdb.set_trace()
        # non-region
        operands = [value_mapper[operand] if operand in value_mapper else operand
                    for operand in self.operands]
        result_types = [o.type for o in self.results]
        successors = [block_mapper[successor] if successor in block_mapper else successor
                    for successor in self.successors]
        attributes = self.attributes.copy()
        properties = self.properties.copy()
        regions = [Region() for _ in range(len(self.regions))]
        # new_op = Operation.create(operands, result_types, successors,
        #                           attributes, properties, regions)
        new_op = self.create(operands, result_types, successors,
                             attributes, properties, regions)

        for i, res in enumerate(new_op.results):
            value_mapper[self.results[i]] = res

        # regions
        for i, region in enumerate(new_op.regions):
            region.clone(self.regions[i], block_mapper, value_mapper)

        return new_op
