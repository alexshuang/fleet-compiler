# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %mlir-opt | %FileCheck %s

def bar():
    a = 10
    b = 11
    return a + b

bar()

# CHECK: module {
# CHECK-NEXT:   func.func @bar() -> i32 {
# CHECK-NEXT:     %c10_i32 = arith.constant 10 : i32
# CHECK-NEXT:     %c11_i32 = arith.constant 11 : i32
# CHECK-NEXT:     %1 = arith.addi %c10_i32, %c11_i32 : i32
# CHECK-NEXT:     return %1 : i32
# CHECK-NEXT:   }
# CHECK-NEXT:   %0 = func.call @bar() : () -> i32
# CHECK-NEXT: }
