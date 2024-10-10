# RUN: fleet_compiler_cli %s | %FileCheck %s

a = 3
b = 40
c = a + b
print(c)

# CHECK: 43
