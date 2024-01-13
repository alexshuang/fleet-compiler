# ===- error.py -------------------------------------------------------------
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

from enum import Enum, auto


class SyntaxErrorCode(Enum):
    Indentation = 1
    Token = auto()
    Colon = auto()
    Comment = auto()
    StringLiteral = auto()
    DecimalLiteral = auto()
    Import = auto()
    Terminator = auto()
    Identifier = auto()
    FunctionDef = auto()
    Statement = auto()
    Parament = auto()
    Argument = auto()
    Type = auto()
    Primary = auto()
    Operator = auto()
    # OpenBracket = auto()            # [
    # CloseBracket = auto()           # ]
    # OpenParen = auto()              # (
    # CloseParen = auto()             # )
    # OpenBrace = auto()              # {
    # CloseBrace = auto()             # }


class SyntaxException(Exception):
    def __init__(self, code, msg) -> None:
        super().__init__(msg)
        self.code = code
