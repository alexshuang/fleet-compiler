# fleet-compiler

An MLIR-based(coming soon) AI compiler designed for PyTorch(coming soon)/Python-like DSL (domain-specific languages) to CPU/RISC-V DSA (domain-specific architectures).

## Quick Start

1. Install from source:
```
pip install fleet-compiler
# Or editable: pip install -e .
```

2. Try one sample, run a "pythonic" example on CPU:
```
fleet_compiler_cli --input examples/ffn_example.py
```

## Developers

```
pip install -r dev_requirements.txt
```

Run tests:
```
pytest tests
```
