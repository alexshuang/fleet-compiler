# RUN: fleet_compiler_cli %s | %FileCheck %s

a = 2
b = 3.0
c = True

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %0 = "arith.constant" () () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %1 = "arith.constant" () () {value = 3.0: f32} : () -> f32
# CHECK-NEXT:   %2 = "arith.constant" () () {value = true} : () -> i1
# CHECK-NEXT: }) {} : () -> ()
