import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, LSTM, Dense

# Load and preprocess the Shakespeare dataset
file_path = "shakespeare.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and target sequences
input_sequences = []
for i in range(1, len(text)):
    seq = text[i - 50:i + 1]  # Use a sequence length of 100 characters
    input_sequences.append(seq)

sequences = tokenizer.texts_to_sequences(input_sequences)
X = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='pre', truncating='pre')
y = tf.keras.utils.to_categorical(X[:, -1], num_classes=total_words)
X = X[:, :-1]

# Build the CNN-LSTM model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=X.shape[1]))
model.add(Conv1D(64, 5, activation='relu'))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=5, batch_size=64)

# Text generation example
seed_text = "To be or not to be, that is the"
for _ in range(100):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=X.shape[1], padding='pre')
    predicted_word_index = tf.argmax(model.predict(sequence), axis=-1).numpy()[0]
    predicted_word = tokenizer.index_word[predicted_word_index]
    seed_text += " " + predicted_word

print(seed_text)
