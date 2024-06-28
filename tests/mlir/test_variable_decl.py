# RUN: fleet_compiler_cli %s | %FileCheck %s

a = 2

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %0 = "arith.constant" () () {value = 2: i32} : () -> i32
# CHECK-NEXT: }) {} : () -> ()
