from fleet_compiler.frontend.python.parsing import Parser
from fleet_compiler.frontend.python.semantic import Pipeline, ReferenceResolvePass, ReplaceAliasOperationNamePass, BuiltinReferenceResolvePass
from fleet_compiler.frontend.python.runtime import Interpreter
from fleet_compiler.frontend.python.error import SyntaxException


def get_basic_pipeline():
    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(ReplaceAliasOperationNamePass())
    pipeline.add(BuiltinReferenceResolvePass())
    return pipeline


def get_interpreter():
    return Interpreter()


def run_test(data: str, pipeline=None, interpreter=None, code=0):
    try:
        m = Parser(data).parse_module()
        if pipeline:
            pipeline.run(m, True)
        if interpreter:
            interpreter.visit(m)
    except SyntaxException as e:
        print(f"error: {code}, {e.code}")
        assert e.code == code
