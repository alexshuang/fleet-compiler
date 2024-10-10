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
from enum import Enum

import warnings
warnings.simplefilter('always', DeprecationWarning)

from fleet_compiler.frontend.lexer import Tokenizer, TokenKind
from fleet_compiler.frontend.parsing import Parser, AstDumper
from fleet_compiler.frontend.semantic import ReferenceResolvePass, OperatorReferenceResolvePass, HandleSliceOpPass
from fleet_compiler.frontend.pass_manager import Pipeline
from fleet_compiler.frontend.ast import AstModule, AstVisitor
from fleet_compiler.frontend.runtime import Interpreter

from fleet_compiler.ir.importer import ASTModuleImporter
from fleet_compiler.ir.pass_manager import PassManager
from fleet_compiler.ir.transforms.lower_numpy import LowerNumpyPass
from fleet_compiler.ir.transforms.inline import InlineFunctionPass
from fleet_compiler.ir.transforms.shape_inference import ShapeInferencePass
from fleet_compiler.ir.transforms.canonicalize import CanonicalizePass
from fleet_compiler.ir.transforms.dce import DeadCodeEliminationPass
from fleet_compiler.ir.transforms.convert_tosa_to_vm import ConvertTosaToVmPass
from fleet_compiler.ir.transforms.convert_tensor_to_vm import ConvertTensorToVmPass
from fleet_compiler.ir.transforms.convert_math_to_vm import ConvertMathToVmPass
from fleet_compiler.ir.transforms.convert_arith_to_vm import ConvertArithToVmPass
from fleet_compiler.ir.transforms.set_target_info import SetTargetInfoPass

from fleet_compiler.vm.bytecode import ByteCodeConverter
from fleet_compiler.vm.vm import VM


class DeprecateAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        self.new_option = kwargs.pop('new_option', None)
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn(
            f"Argument '{option_string}' is deprecated. Use '{self.new_option}' instead.",
            DeprecationWarning,
            )
        if self.new_option:
            new_option_name, new_option_value = self.new_option.split('=')
            setattr(namespace, new_option_name.lstrip('-').replace('-', '_'), new_option_value)


def create_dir(path: str):
    if os.path.isfile(path):
        raise ValueError(f"{path} is an existed file, not directory")
    else:
        os.makedirs(path, exist_ok=True)


def save_output(output_path: str, fn: callable):
    old_stdout = sys.stdout
    with open(output_path, 'w') as fp:
        sys.stdout = fp
        fn()
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


class CompilationPhases(Enum):
    token = 0
    ast = 1
    ir = 2
    mlir = 3
    vm = 4
    bc = 5
    end = 6


def main():
    parser = argparse.ArgumentParser(description='Fleet_compiler_cli: Compile and run python-like code',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='Input file path')
    parser.add_argument('--target-backend', choices=['python', 'sycl', 'cuda', 'llvm-cpu', 'llvm-gpu'],
                        help='Target Backends:\n'
                             '  python (default)\n'
                             '  sycl\n'
                             '  cuda\n'
                             '  llvm-cpu\n'
                             '  llvm-gpu\n')
    parser.add_argument('--compile-to', choices=['token', 'ast', 'ir', 'mlir', 'vm', 'bc'],
                        help='Compilation phases:\n'
                             '  token\n'
                             '  ast\n'
                             '  ir\n'
                             '  mlir\n'
                             '  vm\n'
                             '  bytecode\n')
    parser.add_argument('--runner', choices=['interpreter', 'vm'],
                        default='vm',
                        help='Run with:\n'
                             '  AST interpreter\n'
                             '  vm\n')
    parser.add_argument('--emitToken', action=DeprecateAction, new_option='--compile-to=token', help=argparse.SUPPRESS)
    parser.add_argument('--emitAST', action=DeprecateAction, new_option='--compile-to=ast', help=argparse.SUPPRESS)
    parser.add_argument('--emitMLIR', action=DeprecateAction, new_option='--compile-to=mlir', help=argparse.SUPPRESS)
    parser.add_argument('--emitByteCode', action=DeprecateAction, new_option='--compile-to=bc', help=argparse.SUPPRESS)
    parser.add_argument('--output', '-o', type=str, help='Output file path for emit')
    parser.add_argument('--only-compile', action='store_true', help='not run')
    parser.add_argument('--validation', action='store_true', help='validate the results')
    parser.add_argument('--opt', action=DeprecateAction, new_option='', help=argparse.SUPPRESS)
    parser.add_argument('--dump-intermediates-to', type=str, help='dump *.mlir to path')
    parser.add_argument('--dump-ir-before-pass', action='store_true', help='dump *.mlir')
    parser.add_argument('--dump-ir-after-pass', action='store_true', help='dump *.mlir')
    parser.add_argument('--vm', action=DeprecateAction, new_option='--runner=vm', help=argparse.SUPPRESS)

    # parser.add_argument('--ir-elide-elementsattrs-if-larger', type=int, default=20, help='elide numpy array')
    # np.set_printoptions(args.ir_elide_elementsattrs_if_larger)

    args = parser.parse_args()

    if args.compile_to:
        if args.compile_to == 'token':
            compile_to = CompilationPhases.token
        elif args.compile_to == 'ast':
            compile_to = CompilationPhases.ast
        elif args.compile_to == 'ir':
            compile_to = CompilationPhases.ir
        elif args.compile_to == 'mlir':
            compile_to = CompilationPhases.mlir
        elif args.compile_to == 'vm':
            compile_to = CompilationPhases.vm
        elif args.compile_to == 'bc':
            compile_to = CompilationPhases.bc
    else:
        compile_to = CompilationPhases.end
    
    if intermediates_dir := args.dump_intermediates_to:
        create_dir(intermediates_dir)

    data = open(args.input, "r").read()

    if compile_to == CompilationPhases.token:
        tokenizer = Tokenizer(data)
        token_list = ""
        while tokenizer.peak().kind != TokenKind.EOF:
            tok = tokenizer.next()
            token_list += f"kind={tok.kind.name}, data='{tok.data}'\n"
        if args.output:
            with open(args.output, "w") as fp:
                fp.write(token_list)
        else:
            print(token_list)
        return

    parser = Parser(data)
    ast_module = parser.parse_module()

    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(OperatorReferenceResolvePass())
    pipeline.add(HandleSliceOpPass())
    pipeline.run(ast_module, False)

    if compile_to == CompilationPhases.ast:
        ast_dumper = AstDumper()
        if args.output:
            save_output(args.output, lambda: ast_dumper(ast_module))
        else:
            ast_dumper(ast_module)
        return

    # run AST with interpreter
    if args.runner == 'interpreter':
        interpreter = Interpreter()
        interpreter.visit(ast_module)
        if args.validation:
            if validate(args.input, ast_module, interpreter):
                print("validation success!")
            else:
                print("validation failed!")
        return
    
    module = ASTModuleImporter(ast_module).import_graph()

    if compile_to == CompilationPhases.ir:
        if args.output:
            save_output(args.output, lambda: module.dump())
        else:
            module.dump()
        return

    pm = PassManager(intermediates_dir,
                        args.dump_ir_before_pass, args.dump_ir_after_pass)

    pm.add(InlineFunctionPass())
    pm.add(CanonicalizePass())
    pm.add(DeadCodeEliminationPass())

    pm.add(ShapeInferencePass())
    pm.add(CanonicalizePass())
    pm.add(DeadCodeEliminationPass())

    pm.add(LowerNumpyPass())
    pm.add(CanonicalizePass())
    pm.add(DeadCodeEliminationPass())

    if args.target_backend:
        pm.add(SetTargetInfoPass(args.target_backend))

    if compile_to == CompilationPhases.mlir:
        pm.run(module)
        if args.output:
            save_output(args.output, lambda: module.dump())
        else:
            module.dump()
        return

    pm.add(ConvertArithToVmPass())
    pm.add(ConvertTensorToVmPass())
    pm.add(ConvertMathToVmPass())
    pm.add(ConvertTosaToVmPass())
    pm.add(CanonicalizePass())
    pm.add(DeadCodeEliminationPass())

    pm.run(module)

    if compile_to == CompilationPhases.vm:
        if args.output:
            save_output(args.output, lambda: module.dump())
        else:
            module.dump()
        return

    bc = ByteCodeConverter(module).convert()

    if compile_to == CompilationPhases.bc:
        print(bc)
        return

    if args.only_compile:
        return

    VM(bc).run()


if __name__ == "__main__":
    main()
