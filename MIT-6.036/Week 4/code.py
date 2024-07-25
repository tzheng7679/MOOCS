import numpy as np
import time
# th is a column vector (well, techincally 2d c-vector)

def sd(th, th_0, x, y):
    return np.multiply(
        (np.dot(th.T, x) + th_0) / np.sqrt(np.dot(th.T, th)),
        y
    )

def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])
# #########################
# Section 1
###########################
"""
data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

red_gammas = sd(red_th,red_th0,data,labels)
blue_gammas = sd(blue_th,blue_th0,data,labels)

print("Sums of margins")
print(f"Red: {np.sum(red_gammas)}")
print(f"Blue: {np.sum(blue_gammas)}")

print("Minimum margins")
print(f"Red: {np.min(red_gammas)}")
print(f"Blue: {np.min(blue_gammas)}")

print("Maximum margins")
print(f"Red: {np.max(red_gammas)}")
print(f"Blue: {np.max(blue_gammas)}")
"""

##########################
# Section 3
##########################
"""
def hinge(th,th_0,x,y,gamma_0):
    s = sd(th,th_0,x,y)
    
    return max(0, 1 - s / gamma_0)


data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4

for i in range(data.shape[1]):
    print(hinge(th,th0,data[:,i],labels[:,i],np.sqrt(2)/2))
"""

##########################
# Section 6
##########################

def gd(f, df, x0, step_size_fn, max_iter):
    x = np.copy(x0)
    fs = [f(x0)]
    xs = [x0]
    
    for i in range(max_iter):
        x -= step_size_fn(i) * df(x)
        # fs.append(f(x))
        # xs.append(x)
    
    return (x, fs, xs)

def num_grad(f, delta=0.001):
    def df(x):
        grad = np.empty_like(x)

        for i in range(x.shape[0]):
            d = np.zeros_like(x)
            d[i] += delta

            grad[i, 0] = (f(x + d) - f(x - d)) / (2 * delta)
        return grad
    
    return df

def minimize(f, x0, step_size_fn, max_iter):
    return gd(f, num_grad(f), x0, step_size_fn, max_iter)
"""
def f1(x): return float((2 * x + 3)**2)
def df1(x): return 2 * 2 * (2 * x + 3)
ans=(gd(f1, df1, cv([0.]), lambda i: 0.1, 1000))
print(len(ans[2]))
"""

##########################
# Section 7
##########################
def hinge(v):
    return np.where(v > 1, 0, 1 - v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(np.multiply(
        np.matmul(x.T, th) + th0,
        y.T
    ))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    z = np.mean(hinge_loss(x, y, th, th0)) + lam * np.dot(th.T, th)
    return z[0][0]

########################

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
"""
sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1

x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator
ans = svm_obj(x_1, y_1, th1, th1_0, .1)
# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
"""

###############3
# 7.2
###############3

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v > 1, 0, -1)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return np.multiply(np.multiply(x, d_hinge(np.multiply(
            y.T,
            x.T@th + th0
        )).T), y)
    
# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return np.multiply(d_hinge(np.multiply(
            y.T,
            x.T@th + th0
        )).T, y)

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.reshape(np.mean(d_hinge_loss_th(x, y, th, th0) + 2 * lam * th, axis=1), (th.shape[0], 1))

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.reshape(np.mean(d_hinge_loss_th0(x, y, th, th0), axis=1), (1,1))

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    return np.vstack([
        d_svm_obj_th(X, y, th, th0, lam),
        d_svm_obj_th0(X, y, th, th0, lam)
    ])

# X1 = np.array([[1, 2, 3, 9, 10]])
# y1 = np.array([[1, 1, 1, -1, -1]])
# th1, th10 = np.array([[-0.31202807]]), np.array([[1.834     ]])
# X2 = np.array([[2, 3, 9, 12],
#                [5, 2, 6, 5]])
# y2 = np.array([[1, -1, 1, -1]])
# th2, th20=np.array([[ -3.,  15.]]).T, np.array([[ 2.]])

# # print(d_hinge(np.array([[ 71.]])))
# # print(d_hinge_loss_th(X2[:,0:1], y2[:,0:1], th2, th20))
# # print(d_hinge_loss_th(X2, y2, th2, th20))
# print(d_hinge_loss_th0(X2[:,0:1], y2[:,0:1], th2, th20))
# print(d_hinge_loss_th0(X2, y2, th2, th20))

iterations = 10000
"""
The number of iterations to run on the gradient descent
"""

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    
    def svm_obj_filled(x):
        return svm_obj(data, labels, x[:-1,:], x[-1,:], lam)
    
    def svm_obj_grad_filled(x):
        return svm_obj_grad(data, labels, x[:-1,:], x[-1,:], lam)

    return gd(svm_obj_filled, svm_obj_grad_filled, np.zeros((data.shape[0] + 1, 1)), svm_min_step_size_fn, iterations)

def num_batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    
    def svm_obj_filled(x):
        return svm_obj(data, labels, x[:-1,:], x[-1,:], lam)
    
    def num_grad_filled(x):
        return num_grad(svm_obj_filled)(x)

    return gd(svm_obj_filled, num_grad_filled, np.zeros((data.shape[0] + 1, 1)), svm_min_step_size_fn, iterations)

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

x_1, y_1 = super_simple_separable()

def svm_test(meth):
    t1 = time.time()
    batch = meth(x_1, y_1, 0.0001)
    th = batch[0][:-1,:]
    th0 = batch[0][-1,:]
    tf1 = time.time() - t1
    print(svm_obj(x_1, y_1, th, th0, 0.0001))
    print(tf1)

print("######")
svm_test(batch_svm_min)
print("######")
svm_test(num_batch_svm_min)
# x_1, y_1 = separable_medium()
# print(batch_svm_min(x_1, y_1, 0.0001))