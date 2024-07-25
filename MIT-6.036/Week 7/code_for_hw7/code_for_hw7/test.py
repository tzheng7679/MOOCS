import numpy as np

# def s_max(b):
#     return max(0, b)

# a = np.array([[1, 2, -1], [2, 3, 4]])
# b = map(s_max, a)
# # print(np.vectorize(s_max)(a))

# print(np.e**a)

import code_for_hw7 as hw7

print(hw7.SoftMax.class_fun(hw7.SoftMax, np.array([
    [1, 2, 3],
    [4, 5, 6],
    [-1, 1, 10]
])))