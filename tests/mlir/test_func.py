# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %mlir-opt | %FileCheck %s

import numpy as np

def foo(a, b):
    return a - b

def bar():
    a = 10
    b = 11
    return a + b

bar()

a = np.random.randn(2, 3)
b = np.random.randn(2, 3)
foo(a, b)

# CHECK: module {
# CHECK-NEXT:   func.func @foo(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
# CHECK-NEXT:     %0 = "tosa.sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %0 : tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @bar() -> i32 {
# CHECK-NEXT:     %c10_i32 = arith.constant 10 : i32
# CHECK-NEXT:     %c11_i32 = arith.constant 11 : i32
# CHECK-NEXT:     %0 = arith.addi %c10_i32, %c11_i32 : i32
# CHECK-NEXT:     return %0 : i32
# CHECK-NEXT:   }
# CHECK-NEXT: }
