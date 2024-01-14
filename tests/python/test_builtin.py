import logging
import unittest
from utils import run_e2e_test


class TestBuiltin(unittest.TestCase):
    def testPrint(self):
        data = 'a = 1;print(a)'
        run_e2e_test(data)

    def testAssert(self):
        data = 'a = 1;assert(a == 1)'
        run_e2e_test(data)
        
    def testTime(self):
        data = 'import time;s = time.time();e = time.time() - s;print(e)'
        run_e2e_test(data)
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
