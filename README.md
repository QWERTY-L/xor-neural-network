# xor-neural-network

This is a simple 2-2-1 neural network to solve the XOR problem. You can custumize the activations, number of neurons in the hidden layer etc. This program will use multiple CPU cores to speed up learning.

## Commands

eval [input 1] [input 2] -> uses the NN to predict the Xor using current wieghts. Takes 2 integers

btrain [epochs] -> trains for given amount of epochs. Takes 1 integer.

end -> terminates the program. Takes no inputs.

print [variable] -> prints given variable. Takes 1 string (dont use quotes).

Variables
- x
- y
- w1
- b1
- w2
- b2
- samples
