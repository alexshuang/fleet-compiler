import logging
import unittest

from fleet_compiler.frontend.python.parsing import Parser
from fleet_compiler.frontend.python.semantic import Pipeline, ReferenceResolvePass, ReplaceAliasOperationNamePass, BuiltinReferenceResolvePass
from fleet_compiler.frontend.python.runtime import Interpreter
from fleet_compiler.frontend.python.error import *


def legalize_data(data: str):
    '''
    1. Unindent all statements in the main scope, that is,
       the main scope are not allowed indentation
    2. delete empty lines
    '''
    def get_tab_len(data):
        tab = ""
        for i in range(len(data)):
            if data[i] == '\n':
                i += 1
                while data[i] == ' ':
                    tab += ' '
                    i += 1
                break
        return len(tab)
    
    def the_next_is_empty_line(data):
        for o in data:
            if o != ' ':
                if o == '\n':
                    return True
                else:
                    return False
        return True
    
    res = ""
    indent_phase = True
    tab_len = get_tab_len(data)
    i = 0
    while i < len(data):
        if data[i] == '\n':
            indent_phase = True
            res += data[i]
            i += 1
            if the_next_is_empty_line(data[i:]):
                while i < len(data) - 1:
                    if data[i] != '\n':
                        i += 1
                    else:
                        break
            else:
                if tab_len > 0:
                    # assert data[i:i+tab_len].isspace(), "Unexpected indent"
                    i += tab_len
        elif data[i] != ' ':
            indent_phase = False
            res += data[i]
            i += 1
        else:
            if not indent_phase:
                res += data[i]
            i += 1
            
    return res


def e2e_test(data):
    parser = Parser(legalize_data(data))
    module = parser.parse_module()

    pipeline = Pipeline()
    pipeline.add(ReferenceResolvePass())
    pipeline.add(ReplaceAliasOperationNamePass())
    pipeline.add(BuiltinReferenceResolvePass())
    pipeline.run(module, False)

    interpreter = Interpreter()
    interpreter.visit(module)


def syntax_test(data: str, code=0):
    try:
        parser = Parser(legalize_data(data))
        m = parser.parse_module()
    except SyntaxException as e:
        print(f"error: {code}, {e.code}")
        assert e.code == code


class TestSyntax(unittest.TestCase):
    def testIndent1(self):
        data = '''


        a = 1


        '''
        syntax_test(data)

    def testIndent2(self):
        data = '''
        a = 1


        '''
        syntax_test(data)

    def testIndent3(self):
        data = '''
        a = 1


        '''
        syntax_test(data)

    def testIndent4(self):
        data = '''
        a = 1
        print(a)
        '''
        syntax_test(data, SyntaxErrorCode.Identifier)

    # def testExpression(self):
    #     data = '''
    #     a = 1 * 7 + (2 - 1 + 8**2)
    #     assert(a == 72)
    #     '''
    #     e2e_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
