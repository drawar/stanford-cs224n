import numpy as np
import random

from q3_word2vectest import *
random.seed(314)
predicted = np.random.randn(3)
outputVectors = np.random.randn(10,3)
indices = [4,6,1,7]
cost, gradPred, grad = negSamplingCostAndGradient(predicted, outputVectors, indices)
print cost
print gradPred
print grad