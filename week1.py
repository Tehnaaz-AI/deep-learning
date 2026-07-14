import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Data
x = np.array([[0,0],[0,1],[1,0],[1,1]])
#print(x)
y = np.array([[0],[0],[0],[1]])
print(y)

# Build the Architechture
model = keras.Sequential()
model.add(layers.Input(shape = (2,) )) #evry neuron recives 2 inputs
model.add(layers.Dense(1 , activation = 'sigmoid' ))                                                                                                           


print("Hello World")