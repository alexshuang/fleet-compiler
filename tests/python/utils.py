from fleet_compiler.frontend.parsing import Parser
from fleet_compiler.frontend.semantic import Pipeline, ReferenceResolvePass, OperatorReferenceResolvePass, HandleSliceOpPass
from fleet_compiler.frontend.runtime import Interpreter
from fleet_compiler.frontend.error import SyntaxException


def get_basic_pipeline():
    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(OperatorReferenceResolvePass())
    pipeline.add(HandleSliceOpPass())
    return pipeline


def run_test(data: str, pipeline=None, interpreter=None, code=0):
    try:
        m = Parser(data).parse_module()
        if pipeline:
            pipeline.run(m)
        if interpreter:
            interpreter.visit(m)
    except SyntaxException as e:
        print(f"error: {code}, {e.code}")
        assert e.code == code


def run_e2e_test(data: str, code=0):
    pipeline = get_basic_pipeline()
    interpreter = Interpreter()
    run_test(data, pipeline, interpreter, code)
