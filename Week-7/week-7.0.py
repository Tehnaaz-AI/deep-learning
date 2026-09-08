import tensorflow as tf
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense


# 1. Load IMDB dataset
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Smaller dataset for faster execution
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:3000]
y_test = y_test[:3000]


# 2. Function to create RNN / LSTM / GRU
def create_model(model_type):

    model = Sequential()

    model.add(Embedding(vocab_size, 32))

    if model_type == "RNN":
        model.add(SimpleRNN(32))

    elif model_type == "LSTM":
        model.add(LSTM(32))

    elif model_type == "GRU":
        model.add(GRU(32))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# 3. Train and compare models
results = {}

for name in ["RNN", "LSTM", "GRU"]:

    print("\nTraining", name)

    model = create_model(name)

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2
    )

    training_time = time.time() - start_time

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    results[name] = [accuracy, training_time]


# 4. Print results
print("\nFinal Comparison")

for name in results:
    print(
        name,
        "Accuracy:",
        round(results[name][0], 4),
        "Training Time:",
        round(results[name][1], 2),
        "seconds"
    )


# 5. Accuracy comparison graph
names = list(results.keys())
accuracies = [results[name][0] for name in names]

plt.bar(names, accuracies)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("RNN vs LSTM vs GRU Accuracy")

plt.show()


# 6. Vanishing Gradient demonstration
steps = range(1, 21)

plt.xlabel("Time Steps")
plt.ylabel("Gradient Magnitude")
plt.title("Vanishing Gradient Behaviour")

plt.legend()
plt.show()