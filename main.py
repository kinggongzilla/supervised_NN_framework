import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Input: # needed so layers in Model forward method can be called with prev.output
    def forward(self, inputs):
        self.output = inputs

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., weight_regularizer_l2=0., bias_regularizer_l1=0., bias_regularizer_l2=0.):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
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

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Activation_ReLu:

    def forward(self, inputs):
        self.inputs = inputs  # for calculating derivatives/backpropagation
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = self.output * (1 - self.output) * dvalues

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=-1)

class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss, self.regularization_loss()

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)

        return regularization_loss

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_pred - y_true) ** 2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues) #num samples
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true)/(1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if np.ndim(y_true) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif np.ndim(y_true) == 2:
            correct_confidences = y_pred_clipped * y_true

        return -np.log(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples #normalize gradients


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 # this is the newly calculated derivative of combined softmax with catCrossentropyLoss: y-hat - y
        #normalize gradients
        self.dinputs = self.dinputs / samples



class Optimizer_SDG:
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1+self.decay*self.iterations))

    def update_params(self, layer):
        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = (self.rho * layer.weight_cache) + ((1 - self.rho) * layer.dweights**2)
        layer.bias_cache = (self.rho * layer.bias_cache) + ((1 - self.rho) * layer.dbiases**2)

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1**(1 + self.iterations))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1**(1 + self.iterations))


        layer.weight_cache = (self.beta2 * layer.weight_cache) + ((1 - self.beta2) * layer.dweights**2)
        layer.bias_cache = (self.beta2 * layer.bias_cache) + ((1 - self.beta2) * layer.dbiases**2)

        layer.weight_cache_corrected = layer.weight_cache / (1 - self.beta2**(1 + self.iterations))
        layer.bias_cache_corrected = layer.bias_cache / (1 - self.beta2**(1 + self.iterations))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(layer.weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(layer.bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Accuracy:

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        return accuracy


class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):  # calculates precision value based on passed in true values y
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary=binary

    def init(self, y): #this empty method is only needed since it is called from the train method
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=-1)
        return predictions == y


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_every=1):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            output = self.forward(X)

            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers: # update weights + biases for each layer after every epoch
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f}, ' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}, ' +
                      f'lr: {self.optimizer.current_learning_rate}')

    def finalize(self): # this is needed to be able to loop over layers and call next layers with prev output in worward method
        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                #store which layers are trainable (i.e. have weights)
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output # layer is now the last object from the list

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)



if __name__ == "__main__":

    X, y = sine_data()
    model = Model()

    model.add(Layer_Dense(1, 64))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(64, 64))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(64, 1))
    model.add(Activation_Linear())

    model.set(
        loss=Loss_MeanSquaredError(),
        optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
        accuracy=Accuracy_Regression()
    )

    model.finalize()

    model.train(X, y, epochs=10000, print_every=100)



# if __name__ == "__main__":
#
#     # REGRESSION
#     X, y = sine_data()
#     dense1 = Layer_Dense(1, 64)
#     activation1 = Activation_ReLu()
#     dense2 = Layer_Dense(64, 64)
#     activation2 = Activation_ReLu()
#     dense3 = Layer_Dense(64, 1)
#     activation3 = Activation_Linear()
#     loss_function = Loss_MeanSquaredError()
#     optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
#
#     accuracy_precision = np.std(y) / 250 #250 is random value. --> Higher value => more strickt with accuracy and vice versa
#
#     for epoch in range(10001):
#         #forward
#         dense1.forward(X)
#         activation1.forward(dense1.output)
#         dense2.forward(activation1.output)
#         activation2.forward(dense2.output)
#         dense3.forward(activation2.output)
#         activation3.forward(dense3.output)
#
#         data_loss = loss_function.calculate(activation3.output, y)
#         reg_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)
#         loss = data_loss + reg_loss
#
#         predictions = activation3.output
#         accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision) # checks if difference between prediction and targets is smaller the defined precision
#
#         if epoch % 100 == 0:
#
#             print(f'epoch: {epoch}, ' + f'acc: {accuracy}, ' + f'data_loss: {data_loss}, ' + f'reg_loss: {reg_loss}, ' + f'loss: {loss}, ' + f'lr:  {optimizer.current_learning_rate}')
#
#         #backward
#         loss_function.backward(activation3.output, y)
#         activation3.backward(loss_function.dinputs)
#         dense3.backward(activation3.dinputs)
#         activation2.backward(dense3.dinputs)
#         dense2.backward(activation2.dinputs)
#         activation1.backward(dense2.dinputs)
#         dense1.backward(activation1.dinputs)
#
#         #update params
#         optimizer.pre_update_params()
#         optimizer.update_params(dense1)
#         optimizer.update_params(dense2)
#         optimizer.update_params(dense3)
#         optimizer.post_update_params()
#
#     X_test, y_test = sine_data()
#
#     dense1.forward(X_test)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)
#     dense3.forward(activation2.output)
#     activation3.forward(dense3.output)
#
#     plt.plot(X_test, y_test)
#     plt.plot(X_test, activation3.output)
#     plt.show()

    # LOGISTIC REGRESSION WITH SIGMOID

    # X, y = spiral_data(100, 2)
    #
    # y = y.reshape(-1, 1)
    # dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
    # activation1 = Activation_ReLu()
    # dense2 = Layer_Dense(64, 1)
    # activation2 = Activation_Sigmoid()
    #
    # loss_function = Loss_BinaryCrossEntropy()
    #
    # optimizer = Optimizer_Adam(decay=5e-7)
    #
    # for epoch in range(10001):
    #     dense1.forward(X)
    #     activation1.forward(dense1.output)
    #     dense2.forward(activation1.output)
    #     activation2.forward(dense2.output)
    #
    #     data_loss = loss_function.calculate(activation2.output, y)
    #     regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    #     loss = data_loss + regularization_loss
    #
    #     predictions = (activation2.output > 0.5) * 1
    #     accuracy = np.mean(predictions == y)
    #
    #     if not epoch % 100:
    #         print(f'epoch: {epoch}, ' +
    #               f'acc: {accuracy}, ' +
    #               f'loss: {loss}, ' +
    #               f'data_loss: {data_loss}, ' +
    #               f'regularization loss: {regularization_loss}, ' +
    #               f'lr: {optimizer.current_learning_rate}')
    #
    #     #backwards
    #     loss_function.backward(activation2.output, y)
    #     activation2.backward(loss_function.dinputs)
    #     dense2.backward(activation2.dinputs)
    #     activation1.backward(dense2.dinputs)
    #     dense1.backward(activation1.dinputs)
    #
    #     optimizer.pre_update_params()
    #     optimizer.update_params(dense1)
    #     optimizer.update_params(dense2)
    #     optimizer.post_update_params()
    #
    # #validate model
    # X_test, y_test = spiral_data(100, classes=2)
    #
    # y_test = y_test.reshape(-1, 1)
    #
    # dense1.forward(X_test)
    # activation1.forward(dense1.output)
    # dense2.forward(activation1.output)
    # activation2.forward(dense2.output)
    #
    # loss = loss_function.calculate(activation2.output, y_test)
    #
    # predictions = (activation2.output > 0.5) * 1
    # accuracy = np.mean(predictions == y_test)
    #
    # print(f'validation: acc: {accuracy}, ' +
    #       f'loss: {loss}')


    # CLASSIFICATION WITH SOFTMAX

    # X, y = spiral_data(samples=1000, classes=3)
    # X_test, y_test = spiral_data(samples=100, classes=3)
    #
    # # optimizer = Optimizer_SDG(decay=1e-3,momentum=0.9)
    # # optimizer = Optimizer_Adagrad(decay=1e-5)
    # # optimizer = Optimizer_RMSprop(decay=1e-4)
    # optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
    #
    # dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
    # activation1 = Activation_ReLu()
    # dropout1 = Layer_Dropout(0.1)
    # dense2 = Layer_Dense(512, 3)
    #
    # loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    #
    # for epoch in range(10001):
    #
    #     # forward pass
    #     dense1.forward(X)
    #     activation1.forward(dense1.output)
    #     dropout1.forward(activation1.output)
    #     dense2.forward(dropout1.output)
    #
    #     data_loss = loss_activation.forward(dense2.output, y)
    #     reg_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    #
    #     loss = data_loss + reg_loss
    #
    #     predictions = np.argmax(loss_activation.output, axis=1)
    #
    #     # backward pass
    #     loss_activation.backward(loss_activation.output, y)
    #     dense2.backward(loss_activation.dinputs)
    #     dropout1.backward(dense2.dinputs)
    #     activation1.backward(dropout1.dinputs)
    #     dense1.backward(activation1.dinputs)
    #
    #     # update weights and biases
    #     optimizer.pre_update_params()
    #     optimizer.update_params(dense1)
    #     optimizer.update_params(dense2)
    #     optimizer.post_update_params()
    #
    #     if epoch % 100 == 0:
    #         if len(y.shape) == 2:
    #             y = np.argmax(y, axis=1)
    #         accuracy = np.mean(predictions == y)
    #         print(f'epoch: {epoch}, ' + f'acc: {accuracy}, ' + f'loss: {loss}, ' + f'reg_loss: {reg_loss}')
    #
    # # test data
    # # forward pass
    # dense1.forward(X_test)
    # activation1.forward(dense1.output)
    # dense2.forward(activation1.output)
    #
    # loss = loss_activation.forward(dense2.output, y_test)
    #
    # predictions = np.argmax(loss_activation.output, axis=1)
    #
    # if len(y_test.shape) == 2:
    #     y_test = np.argmax(y_test, axis=1)
    # accuracy = np.mean(predictions == y_test)
    #
    # print('Validation: ' + f'ACC: {accuracy}, ' + f'LOSS: {loss}')



