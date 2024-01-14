import logging
import unittest
from utils import run_e2e_test


class TestNumpy(unittest.TestCase):
    def testNumpy(self):
        data = 'import numpy;assert(numpy.power(3, 3) == 27)'
        run_e2e_test(data)

    def testNp(self):
        data = 'import numpy as np;a = np.power(3, 3);assert(a == 27)'
        run_e2e_test(data)
    
    def testRandSeed(self):
        data = 'import numpy as np;np.random.seed(42);assert(np.allclose(np.random.randn(1), 0.49671415))'
        run_e2e_test(data)

    def testRandn(self):
        data = 'import numpy as np;assert(sum(np.shape(np.random.randn(2, 2))) == 4)'
        run_e2e_test(data)

    def testSqrt(self):
        data = 'import numpy as np;assert(np.sqrt(4) == 2)'
        run_e2e_test(data)
    
    def testMatmul(self):
        data = 'import numpy as np;a = np.random.randn(3, 4);b = np.random.randn(4, 3);c = np.matmul(a, b);print(c)'
        run_e2e_test(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
