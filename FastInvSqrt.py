import math

e = 2 ** 127
dc = math.sqrt(e)
print(e)
print(dc)

d = 1.432430 * 2 ** 63

print(d)
print(dc - d)
print((dc - d)/d)
