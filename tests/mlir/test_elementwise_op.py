# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %mlir-opt | %FileCheck %s

import numpy as np

a = 2
b = 3
c = a + b
c = a - b
c = a * b
c = a / b

a = 1.0
b = 3.0
c = a + b
c = a - b
c = a * b
c = a / b

a = True
b = True
c = a + b
c = a - b
c = a * b
c = a / b

a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = a + b
c = a - b
c = a * b
c = a / b

# CHECK: module {
# CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
# CHECK-NEXT:   %c3_i32 = arith.constant 3 : i32
# CHECK-NEXT:   %0 = arith.addi %c2_i32, %c3_i32 : i32
# CHECK-NEXT:   %1 = arith.subi %c2_i32, %c3_i32 : i32
# CHECK-NEXT:   %2 = arith.muli %c2_i32, %c3_i32 : i32
# CHECK-NEXT:   %3 = arith.divsi %c2_i32, %c3_i32 : i32
# CHECK-NEXT:   %cst = arith.constant 1.000000e+00 : f32
# CHECK-NEXT:   %cst_0 = arith.constant 3.000000e+00 : f32
# CHECK-NEXT:   %4 = arith.addf %cst, %cst_0 : f32
# CHECK-NEXT:   %5 = arith.subf %cst, %cst_0 : f32
# CHECK-NEXT:   %6 = arith.mulf %cst, %cst_0 : f32
# CHECK-NEXT:   %7 = arith.divf %cst, %cst_0 : f32
# CHECK-NEXT:   %true = arith.constant true
# CHECK-NEXT:   %true_1 = arith.constant true
# CHECK-NEXT:   %8 = arith.addi %true, %true_1 : i1
# CHECK-NEXT:   %9 = arith.subi %true, %true_1 : i1
# CHECK-NEXT:   %10 = arith.muli %true, %true_1 : i1
# CHECK-NEXT:   %11 = arith.divsi %true, %true_1 : i1
# CHECK-NEXT:   %cst_2 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
# CHECK-NEXT:   %cst_3 = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
# CHECK-NEXT:   %12 = "tosa.add"(%cst_2, %cst_3) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %13 = "tosa.sub"(%cst_2, %cst_3) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %14 = "tosa.mul"(%cst_2, %cst_3) {shift = 0 : i32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %15 = "tosa.reciprocal"(%cst_3) : (tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %16 = "tosa.mul"(%cst_2, %15) {shift = 0 : i32} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT: }
