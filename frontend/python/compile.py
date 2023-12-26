# ===- compile.py -------------------------------------------------------------
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

import os
import argparse

from lexer import *
from parsing import *
from semantic import *
from runtime import Interpreter


def main():
    parser = argparse.ArgumentParser(description='Compile python into AST/MLIR/bytecode')

    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--emit', type=str, default='token', help='emit Token/AST/MLIR/bytecode')
    # parser.add_argument('--flag', action='store_true', help='A boolean flag')

    args = parser.parse_args()

    output_file = os.path.splitext(args.input)[0] + "." + args.emit.lower()
    data = open(args.input, "r").read()

    tokenizer = Tokenizer(data)

    with open(output_file, "w") as fp:
        while tokenizer.peak().kind != TokenKind.EOF:
            tok = tokenizer.next()
            fp.write(f"kind={tok.kind}, data={tok.data}\n")

    parser = Parser(data)
    module = parser.parse_module()

    print("raw AST:")
    ast_dumper = AstDumper()
    ast_dumper.visit(module)

    print("\nreferenced AST:")
    ref_dumper = RefDumper()
    RefVisitor().visit(module)
    ref_dumper.visit(module)

    print("\nrun:")
    interpreter = Interpreter()
    interpreter.visit(module)


if __name__ == "__main__":
    main()