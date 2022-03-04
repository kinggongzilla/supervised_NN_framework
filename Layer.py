import numpy as np

class Layer_Input: # needed so layers in Model forward method can be called with prev.output
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., weight_regularizer_l2=0., bias_regularizer_l1=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs # remember inputs for backpropagation/calculating derivatives
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #regularization gradients
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            dL2 = 2*self.weights*self.weight_regularizer_l2
            self.dweights += dL2

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.weight_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            dL2 = 2*self.biases*self.bias_regularizer_l2
            self.dbiases += dL2

        #gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    #obsolete, since this is done with optimizer now
    def step(self):
        self.weights = self.weights - self.weights*self.dweights
        self.biases = self.biases - self.biases*self.dbiases

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate # rate = numbers of neurons we want to be 0. self.rate = numbers of neurons we want to normally "fire" (success rate)

    def forward(self, inputs, training):

        self.inputs = inputs

        if not training:
            self.output = inputs.copy

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask