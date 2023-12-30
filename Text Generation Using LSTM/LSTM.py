import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Fetch the content from the URL
url = 'https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt'
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Decode the content assuming it's UTF-8 encoded
    text = response.text
else:
    print(f"Failed to fetch content. Status code: {response.status_code}")
    exit()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
sequences = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(1, len(sequences)):
    n_gram_sequence = sequences[:i+1]
    input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, verbose=2)

# Generate text
seed_text = "hello"
for _ in range(10):
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    padded_seed = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
    predicted_word_index = np.argmax(model.predict(padded_seed), axis=-1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    seed_text += " " + predicted_word

print(seed_text)
