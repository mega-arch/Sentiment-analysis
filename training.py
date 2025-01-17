# Importing necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# Parameters for the sentiment analysis model
VOCAB_SIZE = 10000  # Size of the vocabulary
MAX_LEN = 250  # Maximum length of input sequences
EMBEDDING_DIM = 16  # Dimensionality of the embedding layer
MODEL_PATH = 'sentiment_analysis_model.h5'  # Path to the saved model
file_path = r"C:\Users\santh\Downloads\archive\training.1600000.processed.noemoticon.csv"  # Path to the training data

# Load the dataset and shuffle it
data = pd.read_csv(file_path, encoding='ISO-8859-1')
df_shuffled = data.sample(frac=1).reset_index(drop=True)

# Prepare the texts and labels for training
texts = []
labels = []
for _, row in df_shuffled.iterrows():
    texts.append(row[-1])  # Texts are in the last column
    label = row[0]  # Sentiment labels are in the first column
    labels.append(0 if label == 0 else 1 if label == 2 else 2)  # Mapping sentiment values to 3 classes

texts = np.array(texts)  # Convert list of texts to numpy array
labels = np.array(labels)  # Convert list of labels to numpy array

# Tokenize and pad the sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)  # Fit the tokenizer on the training texts
sequences = tokenizer.texts_to_sequences(texts)  # Convert texts to sequences of integers
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, value=VOCAB_SIZE-1, padding='post')  # Pad sequences to max length

# Save the tokenizer to a pickle file for later use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split the data into training and test sets
train_data = padded_sequences[:-5000]  # All except the last 5000 samples for training
test_data = padded_sequences[-5000:]  # Last 5000 samples for testing
train_labels = labels[:-5000]
test_labels = labels[-5000:]

# Check if a pre-trained model exists
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)  # Load the saved model if it exists
else:
    print("Training a new model...")
    # Define a new sequential model
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),  # Embedding layer
        GlobalAveragePooling1D(),  # Pooling layer
        Dense(16, activation='relu'),  # Dense layer with ReLU activation
        Dense(3, activation='softmax')  # Output layer with softmax activation for 3 classes
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss function
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model for later use
    model.save(MODEL_PATH)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Function to encode text input for prediction
def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)  # Tokenize the text
    tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]  # Convert to word indices
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)  # Pad the sequence

# Interactive loop for real-time predictions
while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ")
    
    if user_input.lower() == 'exit':  # Exit the loop if the user types 'exit'
        break
    
    # Encode the user input and make a prediction
    encoded_input = encode_text(user_input)
    prediction = np.argmax(model.predict(encoded_input))  # Get the sentiment prediction

    # Print the sentiment based on the prediction
    if prediction == 0:
        print("Sentiment: Negative")
    elif prediction == 1:
        print("Sentiment: Neutral")
    else:
        print("Sentiment: Positive")

# Note: The training data can be downloaded from Kaggle: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
