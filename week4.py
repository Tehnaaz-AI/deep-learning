# Write a Python program to build and compare neural networks with different activation functions using Keras. 

# Import Libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Data
(Xtrain,Ytrain),(Xtest,Ytest) = mnist.load_data()
# print(len(Xtrain) , len(Ytrain), len(Xtest), len(Ytest))
print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)

#Build the architecture
activation = ['linear','sigmoid', 'relu', 'tanh', 'softmax', 'leaky_relu']

for i in activation:
    if i == 'leaky_relu':
        model = keras.Sequential([
            layers.Flatten(input_shape=(28,28)),
            layers.Dense(128, activation=layers.LeakyReLU(negative_slope=0.1)),
            layers.Dense(10, activation='softmax')
        ])
    else:
        model = keras.Sequential([
            layers.Flatten(input_shape=(28,28)),
            layers.Dense(128, activation=i),
            layers.Dense(10, activation='softmax')
        ])

    print(f"Model with {i} activation function:")
    model.summary()
    print("\n")



print("Hello Jarvis")