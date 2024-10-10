# RUN: fleet_compiler_cli %s --compile-to=ir | %mlir-opt | %FileCheck %s

a = 2
b = 3.0
c = True
d = [1, 2, 3, 4]
e = [5, 6.0, True, False]
f = a

# CHECK: module {
# CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
# CHECK-NEXT:   %cst = arith.constant 3.000000e+00 : f32
# CHECK-NEXT:   %true = arith.constant true
# CHECK-NEXT:   %cst_0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
# CHECK-NEXT:   %cst_1 = arith.constant dense<[5.000000e+00, 6.000000e+00, 1.000000e+00, 0.000000e+00]> : tensor<4xf32>
# CHECK-NEXT: }