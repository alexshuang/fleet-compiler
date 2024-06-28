# fleet-compiler

An MLIR-based AI compiler designed for Python frontend to Host/RISC-V DSA (domain-specific architectures).

The Fleet compiler parses neural networks written in Python and Numpy into an Abstract Syntax Tree (AST), then translates them into Intermediate Representation (IR).
Optimizations from Tensor to LLVM IR are provided prior to generate RISC-V ISA.

The IR is inspired by the design of MLIR. Its dialects and operations correspond to those in MLIR, with similar syntax and semantics.
It's ensures that any dialect/op can be easily converted into real MLIR, allowing for optimization and code generation by MLIR/LLVM at any compilation stage.

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
pytest tests
```
