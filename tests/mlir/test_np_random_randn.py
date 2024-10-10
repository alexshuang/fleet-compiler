# RUN: fleet_compiler_cli %s --compile-to=ir | %FileCheck %s

import numpy as np

a = np.random.randn(2, 3, 4)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %0 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %1 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %2 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:   %3 = "numpy.random.randn" (%0,%1,%2) : (i32,i32,i32) -> tensor<2x3x4xf32>
# CHECK-NEXT: }) : () -> ()
