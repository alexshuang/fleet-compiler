# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %mlir-opt | %FileCheck %s

a = 2
b = 3
c = a + b

af = 1.0
bf = 3.0
cf = af + bf

ab = True
bb = True
cb = ab + bb

at = [1, 2, 3, 4]
bt = [5, 6, 7, 8]
ct = at + bt

# CHECK: module {
# CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
# CHECK-NEXT:   %c3_i32 = arith.constant 3 : i32
# CHECK-NEXT:   %0 = arith.addi %c2_i32, %c3_i32 : i32
# CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f32
# CHECK-NEXT:   %cst_0 = arith.constant 3.000000e+00 : f32
# CHECK-NEXT:   %1 = arith.addf %cst, %cst_0 : f32
# CHECK-NEXT:   %true = arith.constant true
# CHECK-NEXT:   %true_1 = arith.constant true
# CHECK-NEXT:   %2 = arith.addi %true, %true_1 : i1
# CHECK-NEXT:   %cst_2 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
# CHECK-NEXT:   %cst_3 = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
# CHECK-NEXT:   %3 = "tosa.add"(%cst_2, %cst_3) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT: }