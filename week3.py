# Write a Python program to build and train a neural network using Keras.

from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten 
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#Data
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print(type(X_train),type(Y_train))
print(X_train.shape,Y_train.shape)
plt.imshow(X_train[0])
print(Y_train[0])
plt.show()


#Build the Architecture (MLP)
model = keras.Sequential()
model.add(Flatten(input_shape = (28,28)))
model.add(Dense(10,activation = 'softmax' ))

#Compile
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = keras.optimizers.SGD(learning_rate = 0.001), #mini batch gradient descent,
              metrics = ['accuracy']
              )

#Train
model.fit(X_train, 
          Y_train,
          epochs = 10,
          batch_size = 64
          )

# Task : Improve the train accuracy
# Try : Add more layers

# Build Architecture
model_2 = keras.Sequential()
model_2.add(Flatten(input_shape = (28,28)))
model_2.add(Dense(1024, activation = 'relu'))
model_2.add(Dense(512, activation = 'relu'))
model_2.add(Dense(10, activation = 'softmax'))


# Compile
model_2.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = keras.optimizers.SGD(learning_rate = 0.1), #mini batch gradient descent,
              metrics = ['accuracy']
              )

# Train
history = model_2.fit(X_train, 
          Y_train,
          epochs = 30,
          batch_size = 64,
          validation_data = (X_test, Y_test) 
          )
print(history.history.keys())
print(history.history.values())


# Test Accuracy
test_loss, test_accuracy = model_2.evaluate( X_test , Y_test )
print("Test Loss:",test_loss)
print("Test Accuracy:",test_accuracy)

#Plot
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy', color = 'red')
plt.plot(history.history['accuracy'], label = 'Training Accuracy', color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.plot(history.history['val_loss'], label = 'Validation Loss', color = 'red')
plt.plot(history.history['loss'], label = 'Training Loss', color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()



