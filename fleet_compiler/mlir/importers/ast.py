from fleet_compiler.frontend.python.ast import AstModule, AstVisitor

import iree.compiler
import iree.compiler._mlir_libs

from iree.compiler._mlir_libs._mlir.ir import (
    Module,
    Context,
    Operation,
    NamedAttribute,
    Attribute,
    OpResult,
    OpOperand,
)


class ConvertASTtoMLIR(AstVisitor):
    def __init__(self, cxt) -> None:
        super().__init__()
        self.cxt = cxt

    def run(self, node: AstModule, dump: bool = False):
        m = self.visitModule(node)
        return m
    
    def visitModule(self, node: AstModule):
        return super().visitModule(node)
    


def load_mlir(file_path: str) -> Module:
    try:
        with open(file_path) as fp:
            with Context():
                m = Module.parse(fp.read())
    except Exception as e:
        raise RuntimeError(f"module parse error: {str(e)}")
    return m


class ASTModuleImporter():
    def __init__(self, module: AstModule) -> None:
        self.module = module
    
    def import_graph(self):
        return load_mlir("/work/fleet-compiler/fleet_compiler/mlir/sample.mlir")
        


