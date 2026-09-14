import numpy as np

X = np.array([3.4, 6.7])

y, z = np.split(X, 2)

print(y)
print(z)

a, b = map(float, X)

print(a)
print(b)

c, d = X

print(c)
print(d)

e, f = np.array([8.9, 9.1])

print(e)
print(f)