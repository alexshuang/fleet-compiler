# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %mlir-opt | %FileCheck %s

a = 2
b = 3.0
c = True

# CHECK: module {
# CHECK-NEXT:   %c2_i32 = arith.constant 2 : i32
# CHECK-NEXT:   %cst = arith.constant 3.000000e+00 : f32
# CHECK-NEXT:   %true = arith.constant true
# CHECK-NEXT: }
