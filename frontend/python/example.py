'''
hello world sample
'''
'''
msg = "Hello world!!"

def foo():
   msg = "Hello world!!!"
   print(msg)

def bar():
   return "Hello bar"

foo()
msg = "Hello world!!!"
print(msg)
print(bar())
'''

'''
def foo(msg, msg2=" nested"):
   def bar():
      return "hello bar"

   print(msg, bar())

foo(msg="hello world2")

def foo(msg, msg2=" nested"):
   def bar():
      return "hello bar"

   print(msg, msg2)

msg = "hello world"
foo(msg="hello world2")
foo(msg=msg)
foo(msg)
foo(msg2="not nested", msg=msg)
msg1 = "hello world!!!"
print(msg1)
foo(msg2=msg1, msg=msg)
foo(msg=msg1, msg2=msg)
'''

import numpy as np

np.random.seed(42)
a = np.random.randn(3, 3)
b = np.random.randn(3, 5)
c = np.random.randn(3, 5)
alpha = np.random.randn(3, 5)
beta = np.random.randn(3, 5)
c = a @ b + c
c = alpha * c
c = c + c * 3 - (c / 2 + 3 * np.random.randn(3, 5))
print(c)