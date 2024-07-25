import numpy as np
# 
# 2.8
# 
def index_final_col(A):
    return np.array(A[:,-1].T)

print(index_final_col(np.array([[2, 1], [1, 2]])))