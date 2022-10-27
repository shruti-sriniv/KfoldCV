import numpy as np
import math
import probclearn
import probcpredict
# Input: number of folds k
# numpy matrix X of features, with n rows (samples), d columns (features)
# numpy vector y of scalar values, with n rows (samples), 1 column
# Output: numpy vector z of k rows, 1 column
def run(k,X,y):
    z = np.zeros((k, 1))
    n, d = np.shape(X)
    for i in range(0, k):
        T = set(range(math.floor((n*i)/ k), math.floor((n*(i + 1)) / k)))
        S = set(range(0, n)) - T
        Xtrain = np.zeros((len(S), d))
        Ytrain = np.zeros((len(S), 1))

        counter = 0
        for value in S:
            Ytrain[counter] = y[value]
            Xtrain[counter] = X[value]
            counter = counter + 1

        q, mu_positive, mu_negative, sigma2_positive, sigma2_negative = probclearn.run(Xtrain, Ytrain)
        for t in T:
            if y[t] != probcpredict.run(q, mu_positive, mu_negative, sigma2_positive, sigma2_negative, np.reshape(X[t], (d, 1))):
                z[i] += 1
        z[i] = z[i] / (len(T))
    return z
