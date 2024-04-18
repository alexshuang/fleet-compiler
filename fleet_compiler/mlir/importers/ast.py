from fleet_compiler.frontend.python.ast import (
    AstModule,
    AstVisitor,
    Block as ASTBlock,
    VariableDef,
    IntegerLiteral,
)

from .ir import (
    Module,
    Context,
    SymbolTable,
    FunctionType,
    Location,
    InsertionPoint,
    Attribute,
    Block,
    Operation,
    F32Type,
    RankedTensorType,
    UnrankedTensorType,
    Value,
    IrType,
    IntegerAttr,
    IntegerType,
    func_dialect,
    arith,
)


class ConvertMap:
    def __init__(self) -> None:
        self.cache = {}
    
    def update(self, name: str, fn):
        self.cache[name] = fn
    
    def get(self, name: str):
        return self.cache[name] if name in self.cache else None


def make_constant_op(op_name: str, value_attr: Attribute,
                     result_type = None) -> Operation:
    return Operation.create(
        op_name,
        results=[result_type if result_type else value_attr.type],
        attributes={"value": value_attr},
    )


def integer_attr(value, bits):
    return IntegerAttr.get(IntegerType.get_signless(bits), value)


def integer_type(bits):
    return IntegerType.get_signless(bits)


LITERAL_CONVERT_MAP = ConvertMap()
LITERAL_CONVERT_MAP.update(int, lambda v: make_constant_op(
                                    "arith.constant",
                                    integer_attr(v, 32), integer_type(32)))


class ContextCache:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        with self.ctx:
            self.dtype_to_mlir_type = {
                'int': IrType.parse("i32"),
            }


class ConvertASTtoMLIR(AstVisitor):
    def __init__(self, ast: AstModule) -> None:
        super().__init__()
        self.ast = ast
        self.ctx = Context()
        self.cache = ContextCache(self.ctx)
        self.module = Module.create(self.unknown_loc())
        self.module_ip = InsertionPoint(self.module.body)
        self.op_sym_table = SymbolTable(self.module.operation)
        self.sym_table = {}

    def unknown_loc(self):
        '''
        AST does not provide location
        '''
        return Location.unknown(self.ctx)

    def convert(self):
        return self.visitModule(self.ast)

    def visitModule(self, node: AstModule):
        ftype = FunctionType.get([], [], self.ctx)
        with self.unknown_loc():
            func_op = func_dialect.FuncOp('main', ftype, ip=self.module_ip)
            entry_block = Block.create_at_start(func_op.body, ftype.inputs)
            self.visitBlock(node.block, entry_block)
            self.op_sym_table.insert(func_op)
        return self.module
    
    def visitBlock(self, node: ASTBlock, entry_block: Block):
        with InsertionPoint(entry_block):
            for o in node.stmts:
                self.visit(o)
    
    def visitVariableDef(self, node: VariableDef):
        init_op = self.visit(node.init)
        self.sym_table[node.name] = init_op

    def visitIntegerLiteral(self, node: IntegerLiteral):
        fn = LITERAL_CONVERT_MAP.get(type(node.value))
        return fn(node.value).result


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
        m = ConvertASTtoMLIR(self.module).convert()
        return m
