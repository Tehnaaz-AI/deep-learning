# Build a character-level language model using LSTM for text generation, train 
# on a custom text corpus and generate new text sequences by sampling from 
# the learned probability distribution.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

# Previous version kept for reference:
# # 1. Custom text corpus
# text = """
# artificial intelligence is changing the world.
# deep learning helps computers learn from data.
# neural networks can learn complex patterns.
# """.lower()
#
# # 2. Convert characters to numbers
# chars = sorted(set(text))
#
# char_to_num = {c: i for i, c in enumerate(chars)}
# num_to_char = {i: c for i, c in enumerate(chars)}
#
# # 3. Create sequences
# seq_len = 20
#
# X = []
# y = []
#
# for i in range(len(text) - seq_len):
#     X.append([
#         char_to_num[c]
#         for c in text[i:i + seq_len]
#     ])
#     y.append(char_to_num[text[i + seq_len]])
#
# X = np.array(X)
# y = np.array(y)
#
# # 4. Build LSTM model
# model = Sequential([
#     Input(shape=(seq_len,)),
#     Embedding(len(chars), 16),
#     LSTM(64),
#     Dense(len(chars), activation='softmax')
# ])
#
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy'
# )
#
# # 5. Train
# model.fit(X, y, epochs=30, verbose=1)
#
# # 6. Generate text
# seed = text[:seq_len]
# generated = seed
#
# for i in range(200):
#     x = np.array([
#         [char_to_num[c] for c in seed]
#     ])
#
#     probabilities = model.predict(x, verbose=0)[0]
#     next_num = np.random.choice(len(chars), p=probabilities)
#     next_char = num_to_char[next_num]
#
#     generated += next_char
#     seed = seed[1:] + next_char
#
# print("\nGenerated Text:\n")
# print(generated)

# 1. Custom text corpus
text = """
artificial intelligence is changing the world.
deep learning helps computers learn from data.
neural networks can learn complex patterns.
large language models can generate realistic text.
machine learning systems improve with more examples.
""".lower()

# 2. Convert characters to numbers
chars = sorted(set(text))

char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = {i: c for i, c in enumerate(chars)}

# 3. Create sequences
seq_len = 40

X = []
y = []

for i in range(len(text) - seq_len):
    X.append([
        char_to_num[c]
        for c in text[i:i + seq_len]
    ])
    y.append(char_to_num[text[i + seq_len]])

X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)

# 4. Build a stronger LSTM model
model = Sequential([
    Input(shape=(seq_len,)),
    Embedding(len(chars), 32),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(chars), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train with better batching and validation
history = model.fit(
    X,
    y,
    epochs=80,
    batch_size=64,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

print("\nTraining summary:")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Final val loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")

# 6. Generate text
seed = text[:seq_len]
generated = seed

for i in range(250):
    x = np.array([
        [char_to_num[c] for c in seed]
    ], dtype=np.int32)

    probabilities = model.predict(x, verbose=0)[0]
    temperature = 0.8
    probabilities = probabilities ** (1.0 / temperature)
    probabilities = probabilities / np.sum(probabilities)

    next_num = np.random.choice(len(chars), p=probabilities)
    next_char = num_to_char[next_num]

    generated += next_char
    seed = seed[1:] + next_char

print("\nGenerated Text:\n")
print(generated)