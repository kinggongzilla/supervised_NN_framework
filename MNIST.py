import numpy as np
import pickle
import imageio
from PIL import Image, ImageOps

import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from Layer import Layer_Dense, Layer_Input, Layer_Dropout
from Activation import Activation_Linear, Activation_ReLu, Activation_Softmax, Activation_Sigmoid
from Loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy, Loss_MeanSquaredError, Loss_MeanAbsoluteError, Activation_Softmax_Loss_CategoricalCrossentropy
from Optimizer import Optimizer_Adam, Optimizer_Adagrad, Optimizer_SGD, Optimizer_RMSprop
from Accuracy import Accuracy_Categorical, Accuracy_Regression
from Model import Model
import matplotlib.pyplot as plt

def preprocess(path, invert=True):
    img = Image.open(path).convert('L')
    if invert:
        img = ImageOps.invert(img)
    img.thumbnail((28,28))
    img.save(path)


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,),)])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=transform)
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=transform)

    #prepare for validation test
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    #get 20% of the train set
    split = int(np.floor(0.2 * len(trainset)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    #data loader
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
    validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


    #create model

    model = Model()
    model.add(Layer_Dense(784, 256))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(256, 128))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(128, 64))
    model.add(Activation_ReLu())
    model.add(Layer_Dense(64, 10))
    model.add(Activation_Softmax())

    model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_SGD(learning_rate=0.01, momentum=0.9), accuracy=Accuracy_Categorical())

    model.finalize()
    #
    accuracy_categorical = Accuracy_Categorical()

    #TRAIN MODEL

    for epoch in range(1, 11):

        for train, valid in zip(trainloader, validloader):
            train_images, train_labels = train
            valid_images, valid_labels = valid
            train_images = np.reshape(np.squeeze(train_images), (64, -1)).numpy()
            valid_images = np.reshape(np.squeeze(valid_images), (64, -1)).numpy()

            if train_images.shape[1] == 784: # last batch is not of shape=(64, 784) --> skip
                model.train(train_images, train_labels.numpy(), validation_data=(valid_images, valid_labels.numpy()), epochs=1)


    pickle_out = open("model.pickle", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


    #TESTING
    pickle_in = open("model.pickle", "rb")
    final_model = pickle.load(pickle_in)
    #
    acc = 0
    iterations = 0
    for images, labels in testloader:
        images = np.reshape(np.squeeze(images), (64, -1)).numpy()
        if images.shape[1] == 784:  # last batch is not of shape=(64, 784) --> skip
            iterations += 1
            output = final_model.forward(images, training=False)
            acc += accuracy_categorical.calculate(model.output_layer_activation.predictions(output), labels.numpy())
    print(acc/iterations)


    # preprocess('pullover1.png')

    #INFERENCE
    # tshirt = imageio.imread('tshirt_test.png')
    # tshirt = np.reshape(np.squeeze(tshirt), (1, -1))
    # trouser = imageio.imread('trouser_test.png')
    # trouser = np.reshape(np.squeeze(trouser), (1, -1))
    # sneaker = imageio.imread('sneaker_test.png')
    # sneaker = np.reshape(np.squeeze(sneaker), (1, -1))
    # jacket = imageio.imread('jacket_test.png')
    # jacket = np.reshape(np.squeeze(jacket), (1, -1))
    # bag = imageio.imread('bag_test.png')
    # bag = np.reshape(np.squeeze(bag), (1, -1))
    # sandal = imageio.imread('sandal_test.png')
    # sandal = np.reshape(np.squeeze(sandal), (1, -1))
    # pullover1 = imageio.imread('pullover1.png')
    # pullover1 = np.reshape(np.squeeze(pullover1), (1, -1))
    #
    #
    #
    # prediction = final_model.predict(pullover1)
    #
    #
    #
    # pred_labels = ['Tshirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # for i, label in enumerate(pred_labels):
    #     if prediction == i:
    #         print(label)


    #PLOT SAMPLES

    # dataiter = iter(trainloader)
    # print(dataiter)
    # images, labels = dataiter.next()
    # images = np.squeeze(images)
    #
    # fig = plt.figure(figsize=(15, 5))
    # columns = 5
    # rows = 4
    # for i in range(1, columns*rows + 1):
    #     img = fig.add_subplot(rows, columns, i)
    #     img.imshow(images[i], cmap='gray')
    # plt.show()