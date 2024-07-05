# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

def foo(a, b):
    return a + b

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   func.func @foo() -> () {
# CHECK-NEXT:   }
# CHECK-NEXT: }) : () -> ()
