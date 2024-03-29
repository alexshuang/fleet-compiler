# ===- symbolic.py -------------------------------------------------------------
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

from enum import Enum


class SymbolKind(Enum):
    FunctionSymbol = 1
    VariableSymbol = 2
    OperatorSymbol = 3


class Symbol:
    def __init__(self, kind: SymbolKind, node) -> None:
        self.kind = kind
        self.node = node


class FunctionSymbol(Symbol):
    def __init__(self, kind: SymbolKind, node) -> None:
        super().__init__(kind, node)


class VariableSymbol(Symbol):
    def __init__(self, kind: SymbolKind, node) -> None:
        super().__init__(kind, node)


class OperatorSymbol(Symbol):
    def __init__(self, kind: SymbolKind, op_name: str, node) -> None:
        super().__init__(kind, node)
        self.op_name = op_name
