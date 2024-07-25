import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------
"""
# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
"""
#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

"""
# Get possible combinations of feature and function pairs
feature_combos = []
function_sets = [
    [hw3.raw] * 6,
    [hw3.one_hot] + [hw3.standard] * 4 + [hw3.one_hot]
]
for function_set in function_sets:
    tuple_list = []
    
    for i in range(len(features)):
        feature = features[i][0]
        function = function_set[i]
        tuple_list.append((feature, function))
    
    feature_combos.append(tuple_list)

t_values = [1,10,50]

learners = [hw3.perceptron, hw3.averaged_perceptron]
maxes = ((feature_combos[0], t_values[0], learners[0]), 0)
for feature_combo in feature_combos:
    for T in t_values:
        for learner in learners:
            transformed_data, transformed_labels = hw3.auto_data_and_labels(auto_data_all, feature_combo)
            accu = hw3.xval_learning_alg(learner = learner, data = transformed_data, labels = transformed_labels, k = 10, params = {'T' : T})

            if accu > maxes[1]:
                maxes = ((feature_combo, T, learner), accu)
            # print(f"Accuracy for the {learner.__name__} with T = {T} with functions {feature_combo}")
            # print(accu)
            # print("##___##")

transformed_data, transformed_labels = hw3.auto_data_and_labels(auto_data_all, maxes[0][0])
print(maxes[0][2](data = transformed_data, labels = transformed_labels, params = {"T" : maxes[0][1]}))


feature_combo_test = [('weight', hw3.standard), ('origin', hw3.one_hot)]
transformed_data, transformed_labels = hw3.auto_data_and_labels(auto_data_all, feature_combo_test)
accu = hw3.xval_learning_alg(learner = hw3.averaged_perceptron, data = transformed_data, labels = transformed_labels, k = 10, params = {'T' : 10})
print(accu)
"""
#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------
"""
# Your code here to process the review data
for learner in [hw3.perceptron, hw3.averaged_perceptron]:
    for T in [1, 5, 10]:
        print(learner.__name__)
        print(T)
        print(hw3.xval_learning_alg(learner, review_bow_data, review_labels, 2, {'T': T}))
        print("")
"""
"""
##most positive and negative words
# print(hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, {'T': 10}))
(theta, theta_0) = hw3.averaged_perceptron(review_bow_data, review_labels, {'T': 10})

theta_dict = {i: theta[i] for i in range(len(theta))}
print(theta_dict)
def squared(num):
    return -num
sortedtheta = sorted(theta_dict.values(), key=squared, reverse=True)
print(sortedtheta)

rev_dict = hw3.reverse_dict(dictionary)
for index in range(10):
    print(rev_dict[theta.flatten().tolist().index(sortedtheta[index])])
"""
"""
#most postive and negative
(theta, theta_0) = hw3.averaged_perceptron(review_bow_data, review_labels, {'T': 10})

maxPositive = (0, theta[:,0].dot(review_bow_data[:,0]) + theta_0[0][0])
maxNegative = (0, theta[:,0].dot(review_bow_data[:,0]) + theta_0[0][0])
for reviewIndex in range(review_bow_data.shape[1]):
    review = review_bow_data[:,reviewIndex]
    dist = theta[:,0].dot(review) + theta_0[0][0]

    if dist > maxPositive[1]:
        maxPositive = (reviewIndex, dist)
    if dist < maxNegative[1]:
        maxNegative = (reviewIndex, dist)

print(review_texts[maxPositive[0]])
print("")
print(review_texts[maxNegative[0]])
"""

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[9]["images"]
d1 = mnist_data_all[0]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    x_copy = np.zeros((x.shape[1] * x.shape[2], x.shape[0]))
    for i in range(x.shape[0]):
        x_copy[:,i] = x[i,:,:].flatten()
    return x_copy

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    return np.reshape(np.mean(x, axis=2), (x.shape[0], x.shape[2])).T


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    return np.reshape(np.mean(x, axis=1), (x.shape[0], x.shape[1])).T


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    (n_samples, m, n) = x.shape
    averages = np.zeros((2, n_samples))

    for sample in range(n_samples):
        top_ave = 0
        bottom_ave = 0
        
        for j in range(n):
            for i in range(0, (m)//2): 
                top_ave += x[sample,i,j]
            for i in range((m)//2, m): 
                bottom_ave += x[sample,i,j]

        averages[0, sample] = top_ave / (m//2 * n)
        averages[1, sample] = bottom_ave / ((m+1) // 2 * n)

    return averages

def left_right_features(x):
    """
    Returns a n_sample 2-wide tuples of the left and right averages of a set of images

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    left half of the image = columns 0 to floor(n/2) [exclusive]
    and the second entry is the average of the right half of the image
    = cols floor(n/2) [inclusive] to n
    """
    (n_samples, m, n) = x.shape
    averages = np.zeros((2, n_samples))

    for sample in range(n_samples):
        top_ave = 0
        bottom_ave = 0
        
        for j in range(m):
            for i in range(0, (n)//2): 
                top_ave += x[sample,i,j]
            for i in range((n)//2, n): 
                bottom_ave += x[sample,j,i]

        averages[0, sample] = top_ave / (n//2 * m)
        averages[1, sample] = bottom_ave / ((n+1) // 2 * m)

    return averages
# use this function to evaluate accuracy
# acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

# print(acc)
#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

#raw data
print("Accuracy using raw, flattened data")
print(hw3.get_classification_accuracy(raw_mnist_features(data), labels))


#row average
print("Accuracy using row averaged")
print(hw3.get_classification_accuracy(row_average_features(data), labels))

#column average
print("Accuracy using column averaged")
print(hw3.get_classification_accuracy(col_average_features(data), labels))

#top/bottom
print("Accuracy using top/bottom average")
print(hw3.get_classification_accuracy(top_bottom_features(data), labels))

#left/right
print("Accuracy using top/bottom average")
print(hw3.get_classification_accuracy(left_right_features(data), labels))
