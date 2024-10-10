# RUN: fleet_compiler_cli %s --compile-to=ir | %FileCheck %s

import numpy as np

def foo(a, b):
    return a - b

def bar():
    a = 10
    b = 11
    return a + b

bar()

a = np.random.randn(4)
b = np.random.randn(4)
foo(a, b)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   func.func @foo(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %0 = "tosa.sub" (%arg0,%arg1) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %0: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @bar() -> (i32) {
# CHECK-NEXT:     %2 = "arith.constant" () {value = 10: i32} : () -> i32
# CHECK-NEXT:     %3 = "arith.constant" () {value = 11: i32} : () -> i32
# CHECK-NEXT:     %4 = "arith.addi" (%2,%3) : (i32,i32) -> i32
# CHECK-NEXT:     return %4: i32
# CHECK-NEXT:   }
# CHECK-NEXT:   %6 = "func.call" @bar() : () -> (i32)
# CHECK-NEXT:   %7 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:   %8 = "numpy.random.randn" (%7) : (i32) -> tensor<4xf32>
# CHECK-NEXT:   %9 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:   %10 = "numpy.random.randn" (%9) : (i32) -> tensor<4xf32>
# CHECK-NEXT:   %11 = "func.call" @foo(%8,%10) : (tensor<4xf32>,tensor<4xf32>) -> (tensor<*xf32>)
# CHECK-NEXT: }) : () -> ()
