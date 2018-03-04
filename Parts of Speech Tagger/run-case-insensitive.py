# Extension 3 
from hmm import HiddenMarkovModel
h = HiddenMarkovModel(supervised=True, wordCase='none')
x = h.eval('./pos_test.txt')
print(x)
