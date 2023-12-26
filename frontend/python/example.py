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

def foo(msg):
   print(msg)

msg = "hello world"
foo(msg="hello world2")
