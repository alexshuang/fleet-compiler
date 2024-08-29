# ===- pattern_rewriter.py -------------------------------------------------------------
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

import inspect
from collections.abc import Iterable

from fleet_compiler.ir.builder import InsertionPoint
from .core import *
from .builder import *


class RewritePattern(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> bool: ...


class PatternRewriter:
    def __init__(self) -> None:
        pass

    def replace_op(self, op: Operation, new_results: Sequence[Value] | Value):
        if not isinstance(new_results, list):
            new_results = [new_results]

        new_ops = []
        for v in new_results:
            o = v.owner()
            if o not in new_ops:
                new_ops.append(o)

        self.insert_op_before(op, new_ops)

        assert len(op.results) == len(new_results)
        for old_res, new_res in zip(op.results, new_results):
            self.replace_all_uses_with(old_res, new_res)

    def erase_op(self, op: Operation):
        assert isinstance(block := op.parent, Block)
        block.erase_op(op)

    def insert_op_before(self, op: Operation, new_ops: Sequence[Operation] | Operation):
        assert isinstance(block := op.parent, Block)
        new_ops = [new_ops] if not isinstance(new_ops, Iterable) else new_ops
        for o in new_ops:
            o.parent = None
            block.insert_before(o, op)

    def replace_all_uses_with(self, old: Value, new: Value):
        old.replace_by(new)


class RewritePatternApplier:
    def __init__(self, pattern_set: list[RewritePattern], reverse_ops=False):
        self.rewriter = PatternRewriter()
        self.pattern_set = pattern_set
        self.reverse_ops = reverse_ops

    def apply(self, op: Operation):
        for p in self.pattern_set:
            worklist = self._populate_worklist(op)
            while worklist:
                _op = worklist.pop(0)
                if p.match_and_rewrite(_op, self.rewriter):
                    worklist = self._populate_worklist(op)

    def _populate_worklist(self, op: Operation):
        return [o for o in op.walk(self.reverse_ops)]


def op_rewrite_pattern(func: callable[RewritePattern, Operation, PatternRewriter]):
    params = [o for o in inspect.signature(func).parameters.values()]
    expected_type = params[1].annotation

    def impl(self, op: Operation, rewriter: PatternRewriter):
        if type(op) is expected_type:
            return func(self, op, rewriter)
        return False

    return impl
