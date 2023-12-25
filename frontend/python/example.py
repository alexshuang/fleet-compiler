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

msg = "hello world"

# def foo(msg = "hello world"):
def foo(msg=msg,):
   print(msg)

foo()
