# Script to get the word level accuracy of the HMM trained on pos_train.txt
from hmm import HiddenMarkovModel
h = HiddenMarkovModel(supervised=False)
h.train()
x = h.eval('./pos_test.txt')
print(x)
