import numpy as np

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# w = np.array([[1, -1, -2], [-1, 2, 1]])
# x = np.array([[1, 1]]).T
# y = np.array([[0, 1, 0]]).T

# z = w.T@x
# a = softmax(z)
# grad = np.dot(x, (a - y).T)

# w_star = w - .5 * grad

# print(softmax(w_star.T@x))

w = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]])
x = np.array([[2, 13, 0, 0]]).T

z = w@x + np.array([[0, 2]]).T
a = softmax(z)

print(a)