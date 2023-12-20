import os
import argparse

from lexer import *
from parsing import *
from semantic import *


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

    ref_dumper = RefDumper()
    ref_dumper.visit(module)

    RefVisitor().visit(module)
    ref_dumper.visit(module)


if __name__ == "__main__":
    main()