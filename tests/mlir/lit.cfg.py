import os
import lit

config.name = "example"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.py']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, '/tmp')
config.excludes = ['lit.cfg.py', 'test_lit.py']
config.substitutions.append(['%FileCheck', '/usr/lib/llvm-15/bin/FileCheck'])
config.substitutions.append(['%mlir-opt', 'mlir-opt-15'])
