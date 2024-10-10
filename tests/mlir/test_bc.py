# RUN: fleet_compiler_cli %s --compile-to=bc | %FileCheck %s

a = 3
b = 40
c = a + b
print(c)

# CHECK: ByteCodeModule(consts=[40, 'device_add_sig_ii_i_ins_i32_i32_outs_i32', 'python_print_sig_i_ins_i32'], code=[<OpCode.iconst_3: 5>, <OpCode.ldc: 11>, <TypeCode.scalar: 1>, 0, <OpCode.invokestatic: 12>, 1, <OpCode.istore_2: 21>, <OpCode.iload_2: 16>, <OpCode.invokestatic: 12>, 2], target='python', variable_size=3)
