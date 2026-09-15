# Implement a Denoising Autoencoder on MNIST, 
# compare reconstruction quality on noisy inputs, and 
# visualize the 2D latent space to analyze how the model organizes digit representations.

import numpy as np 
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# Dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# Flatten
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
print(x_train.shape)

# Normalize
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(x_train[0])

# Add Noise
xtrain_noisy = x_train + 0.4* (np.random.normal(size = x_train.shape))
print(len(xtrain_noisy))
xtest_noisy = x_test + 0.4* (np.random.normal(size = x_test.shape))

x_train = np.clip(x_train,0,1)
x_test = np.clip(x_test,0,1)

# Build the Architecture
encoder = keras.Sequential()
encoder.add(layers.Input(shape = (784,)))
encoder.add(layers.Dense(512, activation= 'relu'))
encoder.add(layers.Dense(128, activation= 'relu'))
encoder.add(layers.Dense(2, activation = 'linear'))

decoder = keras.Sequential()
decoder.add(layers.Input(shape = (2,)))
decoder.add(layers.Dense(128, activation= 'relu'))
decoder.add(layers.Dense(512, activation= 'relu'))
decoder.add(layers.Dense(784, activation= 'sigmoid'))

autoencoder = keras.Sequential([encoder, decoder])

# Compile
autoencoder.compile(
    optimizer = 'adam',
    loss = 'mse'
)

# Train
autoencoder.fit(xtrain_noisy, x_train,
                batch_size = 128,
                epochs = 5
                )

# Predict
predictions = autoencoder.predict(xtest_noisy)
# plt.imshow(predictions[0].reshape(28,28), cmap='gray')

latent_values = encoder.predict(xtest_noisy)

# Visualization
for i in range(5):
    plt.subplot(3,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap = 'gray' )
    plt.title('Original')
    
    plt.subplot(3,5,i+6)
    plt.imshow(xtest_noisy[i].reshape(28,28), cmap = 'gray' )
    plt.title('Noisy')

    plt.subplot(3,5,i+11)
    plt.imshow(predictions[i].reshape(28,28), cmap = 'gray' )
    plt.title('Reconstructed')

plt.show()


# Compare construction quality
noisy_mse = np.mean((x_test - xtest_noisy ) ** 2)
reconstrcuted_mse = np.mean((x_test - predictions ) ** 2)
print("Noisy MSE = ",noisy_mse)
print("Reconstructed MSE = ",reconstrcuted_mse)

# Visualize
plt.scatter(latent_values[:,0],
            latent_values[:,1],
            c = y_test,
            cmap = 'tab10',
            s = 5   )
# plt.colorbar(scatter, label= "digitclause")
plt.xlabel('latent_dimension1')
plt.ylabel('latent_dimension2')
plt.title('2D Latent Space of MNIST')
plt.show()



   







































