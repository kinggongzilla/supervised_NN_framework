import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
from Layer import Layer_Dense, Layer_Input, Layer_Dropout
from Activation import Activation_Linear, Activation_ReLu, Activation_Softmax, Activation_Sigmoid
from Loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy, Loss_MeanSquaredError, Loss_MeanAbsoluteError, Activation_Softmax_Loss_CategoricalCrossentropy
from Optimizer import Optimizer_Adam, Optimizer_Adagrad, Optimizer_SDG, Optimizer_RMSprop
from Accuracy import Accuracy_Categorical, Accuracy_Regression
from Model import Model
import matplotlib.pyplot as plt

nnfs.init()

if __name__ == "__main__":

    # WITH DROPOUT
    X, y = spiral_data(samples=100, classes=2)
    X_test, y_test = spiral_data(samples=100, classes=2)

    model = Model()
    model.add(Layer_Dense(2, 512))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(512, 3))
    model.add(Activation_Softmax())

    model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5), accuracy=Accuracy_Categorical())

    model.finalize()

    model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

    #LOGISTIC REGRESSION
    # X, y = spiral_data(samples=100, classes=2)
    # X_test, y_test = spiral_data(samples=100, classes=2)
    # y = y.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)
    #
    # model = Model()
    # model.add(Layer_Dense(2, 64))
    # model.add(Activation_ReLu())
    # model.add(Layer_Dense(64, 1))
    # model.add(Activation_Sigmoid())
    #
    # model.set(loss=Loss_BinaryCrossEntropy(), optimizer=Optimizer_Adam(), accuracy=Accuracy_Categorical(binary=True))
    #
    # model.finalize()
    #
    # model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

    #REGRESSION
    # X, y = sine_data()
    # model = Model()
    #
    # model.add(Layer_Dense(1, 64))
    # model.add(Activation_ReLu())
    # model.add(Layer_Dense(64, 64))
    # model.add(Activation_ReLu())
    # model.add(Layer_Dense(64, 1))
    # model.add(Activation_Linear())
    #
    # model.set(
    #     loss=Loss_MeanSquaredError(),
    #     optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    #     accuracy=Accuracy_Regression()
    # )
    #
    # model.finalize()
    #
    # model.train(X, y, epochs=10000, print_every=100)



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



