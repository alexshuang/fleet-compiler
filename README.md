# fleet-compiler

An MLIR-based AI compiler designed for Python-like and PyTorch frontends to CPU/RISC-V DSA (domain-specific architectures).

_Note: Python-like E2E will be supported in preference to PyTorch._

## Quick Start

1. Install from source:
```
pip install fleet-compiler
# Or editable: pip install -e .
```

2. Try inference a "Pythonic" gpt2 example:
```
fleet_compiler_cli --input examples/gpt2.py
```

## Developers

```
pip install -r dev_requirements.txt
```

Run tests:
```
pytest tests
```
