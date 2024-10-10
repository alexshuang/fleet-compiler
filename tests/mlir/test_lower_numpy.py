# RUN: fleet_compiler_cli %s --compile-to=mlir | %FileCheck %s

import numpy as np

x = 20.0
x2 = [20.0, 23.0, 43.0]
b = np.sqrt(x)
b2 = np.sqrt(x2)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %0 = "arith.constant" () {value = 20.0: f32} : () -> f32
# CHECK-NEXT:   %1 = "arith.constant" () {value = dense<[20.0, 23.0, 43.0]>: tensor<3xf32>} : () -> tensor<3xf32>
# CHECK-NEXT:   %4 = "math.sqrt" (%0) : (f32) -> f32
# CHECK-NEXT:   %5 = "math.sqrt" (%1) : (tensor<3xf32>) -> tensor<3xf32>
# CHECK-NEXT: }) : () -> ()
