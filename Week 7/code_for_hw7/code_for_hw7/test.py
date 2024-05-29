import numpy as np

def s_max(b):
    return max(0, b)

a = np.array([1, 2, -1])
b = map(s_max, a)
print(np.vectorize(s_max)(a))
