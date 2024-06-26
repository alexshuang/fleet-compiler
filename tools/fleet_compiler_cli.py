# ===- fleet_compiler_cli.py -------------------------------------------------------------
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

import io
import os
import sys
import subprocess
import argparse
from typing import Union
import numpy as np

from fleet_compiler.frontend.python.lexer import Tokenizer, TokenKind
from fleet_compiler.frontend.python.parsing import Parser, AstDumper
from fleet_compiler.frontend.python.semantic import ReferenceResolvePass, OperatorReferenceResolvePass, HandleSliceOpPass
from fleet_compiler.frontend.python.pass_manager import Pipeline
from fleet_compiler.frontend.python.ast import AstModule, AstVisitor
from fleet_compiler.frontend.python.runtime import Interpreter

from fleet_compiler.ir.importer import ASTModuleImporter
from fleet_compiler.ir.printer import Printer

def create_dir(path: str):
    if os.path.isfile(path):
        raise ValueError(f"{path} is an existed file, not directory")
    elif not os.path.isdir(path):
        os.makedirs(path)


def save_output(output_path: str, m: AstModule, v: AstVisitor):
    old_stdout = sys.stdout
    with open(output_path, 'w') as fp:
        sys.stdout = fp
        v.visit(m)
    sys.stdout = old_stdout


def allmost_equal(a, b):
    if type(a) != type(b):
        return False
    
    if isinstance(a, Union[list, tuple]):
        if len(a) != len(b):
            return False
        for _a, _b in zip(a, b):
            return allmost_equal(_a, _b)
    elif isinstance(a, np.ndarray):
        return np.allclose(a, b)
    else:
        return a == b


def validate(src_path: str, m: AstModule, v: AstVisitor):
    # Run src py as a subprocess
    process = subprocess.Popen(['python', src_path], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    ref, error = process.communicate()
    # Check for errors
    if process.returncode != 0:
        print(f"Error running {src_path}: {error}")
        sys.exit(1)

    # redirect output
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    v.visit(m)
    sys.stdout = old_stdout

    return allmost_equal(eval(output.getvalue()), ref)
    

def main():
    parser = argparse.ArgumentParser(description='Compile python into AST/MLIR/bytecode')

    parser.add_argument('input', type=str, required=True, help='Input file path')
    parser.add_argument('--output-dir', type=str, help='Output file path')
    parser.add_argument('--emitToken', action='store_true', help='emit *.token')
    parser.add_argument('--emitAST', action='store_true', help='emit *.ast')
    parser.add_argument('--emitMLIR', action='store_true', help='emit *.mlir')
    parser.add_argument('--emitByteCode', action='store_true', help='emit *.bc')
    parser.add_argument('--only-compile', action='store_true', help='not run')
    parser.add_argument('--validation', action='store_true', help='validate the results')

    args = parser.parse_args()

    if args.output_dir:
        create_dir(args.output_dir)
    else:
        args.output_dir = "."

    output_file_stem = os.path.join(args.output_dir, os.path.splitext(args.input)[0])
    data = open(args.input, "r").read()

    if args.emitToken:
        tokenizer = Tokenizer(data)
        output_file = output_file_stem + ".token"
        with open(output_file, "w") as fp:
            while tokenizer.peak().kind != TokenKind.EOF:
                tok = tokenizer.next()
                fp.write(f"kind={tok.kind.name}, data={tok.data}\n")

    parser = Parser(data)
    ast_module = parser.parse_module()

    ast_dumper = AstDumper()

    print("raw AST:")
    ast_dumper.visit(ast_module)

    if args.emitAST:
        save_output(output_file_stem + ".before.ast", ast_module, ast_dumper)

    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(OperatorReferenceResolvePass())
    pipeline.add(HandleSliceOpPass())
    pipeline.run(ast_module, True)

    if args.emitAST:
        save_output(output_file_stem + ".after.ast", ast_module, ast_dumper)

    if args.only_compile:
        return

    print("\nraw IR:")
    module = ASTModuleImporter(ast_module).import_graph()
    module.dump()

    print("\nrun:")
    interpreter = Interpreter()
    interpreter.visit(ast_module)

    if args.validation:
        if validate(args.input, ast_module, interpreter):
            print("validation success!")
        else:
            print("validation failed!")


if __name__ == "__main__":
    main()
