# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

import numpy as np

a = 2
b = 3
c = a + b
c = a - b
c = a * b
c = a / b
c = a ** b

a = 1.0
b = 3.0
c = a + b
c = a - b
c = a * b
c = a / b
c = a ** b

a = 2.0
b = 4
c = a ** b

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
c = a ** b

a = np.random.randn(2, 2, 3)
b = np.random.randn(3, 4)
c = a @ b

a = np.random.randn(2, 3)
c = a @ b

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %0 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %1 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %2 = "arith.addi" (%0,%1) : (i32,i32) -> i32
# CHECK-NEXT:   %3 = "arith.subi" (%0,%1) : (i32,i32) -> i32
# CHECK-NEXT:   %4 = "arith.muli" (%0,%1) : (i32,i32) -> i32
# CHECK-NEXT:   %5 = "arith.divsi" (%0,%1) : (i32,i32) -> i32
# CHECK-NEXT:   %6 = "math.ipowi" (%0,%1) : (i32,i32) -> i32
# CHECK-NEXT:   %7 = "arith.constant" () {value = 1.0: f32} : () -> f32
# CHECK-NEXT:   %8 = "arith.constant" () {value = 3.0: f32} : () -> f32
# CHECK-NEXT:   %9 = "arith.addf" (%7,%8) : (f32,f32) -> f32
# CHECK-NEXT:   %10 = "arith.subf" (%7,%8) : (f32,f32) -> f32
# CHECK-NEXT:   %11 = "arith.mulf" (%7,%8) : (f32,f32) -> f32
# CHECK-NEXT:   %12 = "arith.divf" (%7,%8) : (f32,f32) -> f32
# CHECK-NEXT:   %13 = "math.powf" (%7,%8) : (f32,f32) -> f32
# CHECK-NEXT:   %14 = "arith.constant" () {value = 2.0: f32} : () -> f32
# CHECK-NEXT:   %15 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:   %16 = "math.fpowi" (%14,%15) : (f32,i32) -> f32
# CHECK-NEXT:   %17 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %18 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %19 = "arith.addi" (%17,%18) : (i1,i1) -> i1
# CHECK-NEXT:   %20 = "arith.subi" (%17,%18) : (i1,i1) -> i1
# CHECK-NEXT:   %21 = "arith.muli" (%17,%18) : (i1,i1) -> i1
# CHECK-NEXT:   %22 = "arith.divsi" (%17,%18) : (i1,i1) -> i1
# CHECK-NEXT:   %23 = "arith.constant" () {value = dense<[1, 2, 3, 4]>: tensor<4xi32>} : () -> tensor<4xi32>
# CHECK-NEXT:   %24 = "arith.constant" () {value = dense<[5, 6, 7, 8]>: tensor<4xi32>} : () -> tensor<4xi32>
# CHECK-NEXT:   %25 = "tosa.add" (%23,%24) : (tensor<4xi32>,tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %26 = "tosa.sub" (%23,%24) : (tensor<4xi32>,tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %27 = "tosa.mul" (%23,%24) {shift = 0: i32} : (tensor<4xi32>,tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %28 = "tosa.reciprocal" (%24) : (tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %29 = "tosa.mul" (%23,%28) {shift = 0: i32} : (tensor<4xi32>,tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %30 = "math.ipowi" (%23,%24) : (tensor<4xi32>,tensor<4xi32>) -> tensor<4xi32>
# CHECK-NEXT:   %31 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %32 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %33 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %34 = "numpy.random.randn" (%31,%32,%33) : (i32,i32,i32) -> tensor<2x2x3xf32>
# CHECK-NEXT:   %35 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %36 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:   %37 = "numpy.random.randn" (%35,%36) : (i32,i32) -> tensor<3x4xf32>
# CHECK-NEXT:   %38 = "tosa.reshape" (%34) {new_shape = array<i32: 4, 3>} : (tensor<2x2x3xf32>) -> tensor<4x3xf32>
# CHECK-NEXT:   %39 = "tosa.matmul" (%38,%37) : (tensor<4x3xf32>,tensor<3x4xf32>) -> tensor<4x4xf32>
# CHECK-NEXT:   %40 = "tosa.reshape" (%39) {new_shape = array<i32: 2, 2, 4>} : (tensor<4x4xf32>) -> tensor<2x2x4xf32>
# CHECK-NEXT:   %41 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %42 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %43 = "numpy.random.randn" (%41,%42) : (i32,i32) -> tensor<2x3xf32>
# CHECK-NEXT:   %44 = "tosa.matmul" (%43,%37) : (tensor<2x3xf32>,tensor<3x4xf32>) -> tensor<2x4xf32>
# CHECK-NEXT: }) : () -> ()
