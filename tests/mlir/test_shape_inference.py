# RUN: fleet_compiler_cli %s --emitMLIR --only-compile --opt | %FileCheck %s

import numpy as np

def foo(x):
    b = np.sqrt(x)
    return b

x = 20.0
b = foo(x)
print(b)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %2 = "arith.constant" () {value = 20.0: f32} : () -> f32
# CHECK-NEXT:   %6 = "tensor.cast" (%2) : (f32) -> f32
# CHECK-NEXT:   %7 = "math.sqrt" (%6) : (f32) -> f32
# CHECK-NEXT:   "python.print" (%7) : (f32) -> ()
# CHECK-NEXT: }) : () -> ()

