import numpy as np
import code_for_hw5 as hw5
import os
#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw5.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw5.standard and hw5.one_hot.
# 'name' is not numeric and would need a different encoding.
features1 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

features2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = hw5.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = hw5.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = hw5.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset
max = ((0, 0, 1), hw5.xval_learning_alg(auto_data[0], auto_values, 0, 10)[0][0] * sigma)

# Checking what minimizes RMSE out of ALL feature sets, polynomial orders, and lambdas
"""
for featureSet in range(2):
    for lam in np.arange(stop=.11, step=.01):
        for poly in range(1, 4):
            print((featureSet, lam, poly))
            features = hw5.make_polynomial_feature_fun(poly)(auto_data[featureSet])
            score = hw5.xval_learning_alg(features, auto_values, lam, 10) * sigma

            print(score)
            if score[0][0] < max[1]:
                max = ((featureSet, lam, poly), score[0][0])
"""
# Checking what minimizes for first features set of order 3
polyfunc = hw5.make_polynomial_feature_fun(3)
features = polyfunc(auto_data[0])
min = (0, hw5.xval_learning_alg(features, auto_values, 0, 10) * sigma)

for lam in np.arange(stop=100, step=2):
    score = hw5.xval_learning_alg(polyfunc(auto_data[0]), auto_values, lam, 10) * sigma
    if score < min[1]:
        min = (lam, score)

print(min)