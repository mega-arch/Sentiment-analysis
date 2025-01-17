# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Constants for vocabulary size, maximum length of input sequences, and model path
VOCAB_SIZE = 10000
MAX_LEN = 250
MODEL_PATH = 'sentiment_analysis_model.h5'

# Load the saved sentiment analysis model from the file
model = load_model(MODEL_PATH)

# Load the tokenizer that was used during model training to convert words to tokens
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to encode a list of texts into sequences of integers
def encode_texts(text_list):
    encoded_texts = []  # List to store the encoded text sequences
    for text in text_list:
        # Convert text into a list of word tokens
        tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
        # Convert tokens into indices based on the tokenizer's word index (or 0 if the word is not found)
        tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
        encoded_texts.append(tokens)  # Append the tokenized sequence to the list
    # Pad the sequences to ensure they are of uniform length (MAX_LEN) and return the padded sequences
    return pad_sequences(encoded_texts, maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

# Function to predict sentiment for a list of texts
def predict_sentiments(text_list):
    # Encode the input texts into integer sequences
    encoded_inputs = encode_texts(text_list)
    # Get the model predictions for the encoded inputs
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    
    sentiments = []  # List to store the sentiment labels for each prediction
    for prediction in predictions:
        # Assign sentiment labels based on the predicted class
        if prediction == 0:
            sentiments.append("Negative")  # Class 0 corresponds to Negative sentiment
        elif prediction == 1:
            sentiments.append("Neutral")   # Class 1 corresponds to Neutral sentiment
        else:
            sentiments.append("Positive")  # Class 2 corresponds to Positive sentiment
    return sentiments
