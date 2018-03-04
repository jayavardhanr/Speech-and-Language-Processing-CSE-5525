# Script to get word level accuracy of the hidden markov model with the most probable
# tag counts 
from hmm import HiddenMarkovModel
h = HiddenMarkovModel(supervised=True)
x = h.eval('./pos_test.txt')
print(x)
