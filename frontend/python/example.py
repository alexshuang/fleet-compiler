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

def foo(msg = "hello world"):
   print(msg)

foo()
