# fleet-compiler

Fleet compiler is based on MLIR and aims to compile Python frontend code to Host/RISC-V domain-specific architectures (DSA).

The compiler parses neural networks written in Python and Numpy into Abstract Syntax Tree (AST), translates into Intermediate Representation (IR) for graph optimiaztion.

The IR is inspired by MLIR, with its dialects and operations having similar syntax and semantics to those in MLIR. This ensures that any dialect and operation can be easily converted into MLIR, allowing for optimization and code generation by MLIR/LLVM.

## Quick Start

1. Install from source:
```
pip install fleet-compiler
```

2. Try to inference gpt2:
```
fleet_compiler_cli examples/gpt2.py
```

3. Dump executable intermediaes AST/MLIR:
```
fleet_compiler_cli examples/gpt2.py --emitAST --emitMLIR --only-compile
```

## Developers

```
pip install -r dev_requirements.txt
apt install llvm-15 mlir-tools-15
```

Run tests:
```
pytest tests/python
lit tests/mlir
```
