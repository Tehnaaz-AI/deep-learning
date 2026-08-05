import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Data
x = np.array([[0,0],[0,1],[1,0],[1,1]])
#print(x)
y = np.array([[0],[0],[0],[1]])
# print(y)

# Build the Architecture
model = keras.Sequential()
model.add(layers.Input(shape = (2,) )) #every neuron receives 2 inputs
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





print("Hello World")