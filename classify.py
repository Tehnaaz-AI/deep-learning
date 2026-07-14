import numpy as np
from sklearn.datasets import load_iris
from tensorflow import keras
from tensorflow.keras import layers

# Data
data = load_iris()
print(data)
x = data.data
y = data.target

print(x.shape)




"""
# Build the Architechture
model = keras.Sequential()
model.add(layers.Input(shape = (2,) )) #evry neuron recives 2 inputs
model.add(layers.Dense(1 , activation = 'sigmoid' ))  
#model.add(layers.Dense(4,))                                                                                                         
model.summary()

#compile
model.compile( loss='mse' ,  
    optimizer = keras.optimizers.SGD(learning_rate = 0.1),
    metrics = ['accuracy'])

#train
model.fit(x,
          y,
          epochs = 300)

loss = model.evaluate( x , y )
print(loss)
"""