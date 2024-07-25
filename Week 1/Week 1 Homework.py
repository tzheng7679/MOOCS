import numpy as np

def length(v):
    """
    returns the magnitude of the vector (i.e. ||v||)
    """
    return np.array([[np.linalg.norm(v)]])

def signed_dist(x, th, th0):
    """
    returns the orthogonal distance of point [x] (expressed as a column vector) and the hyper plane with th expressed as a row vector
    """
    return (np.dot(x.T, th) + th0) / length(th)

def positive(x, th, th0):
    """
    :param
    returns a matrix where row i corresponds to the ith "th" value provided
    """
    return np.sign(((np.matmul(th.T, x) + th0) / length(th)))


def score1(x, target, th, th0):
    """
    returns the score of a hyperplane on a set of data
    """
    return np.sum(positive(x, th, th0) == target, axis=1)

def prob141(data, th, th0):
    return positive(data, th, th0)

data = np.transpose(np.array([[1, 2], [1, 3], [2, 1], [1, -1], [2, -1]]))
th_0 = np.array([[1, 0]])
th_1 = np.array([[0, 1]])
th_2 = np.array([[1, 1]])
ths = np.concatenate((th_0, th_1, th_2), axis = 0).T
th0_0s = np.array([[0]])
th0_1s = np.array([[0]])
th0_2s = np.array([[-1]])
th0s = np.concatenate((th0_0s, th0_1s, th0_2s))

target = np.array([[1, 1, 1]]).T

print(score1(data, target, ths, th0s))