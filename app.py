import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if not token_list:  # Handle out-of-vocabulary words
        return "Unknown word (not in vocabulary)"

    # Ensure token_list matches model input shape
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    predicted_word = tokenizer.index_word.get(predicted_word_index, "Unknown word")  # Safe lookup

    return predicted_word

# Streamlit app
st.title("Next Word Prediction With LSTM + Attention")
input_text = st.text_input("Enter the sequence of Words", "To be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]  # Use correct input shape
    st.write(f"Max sequence length used: {max_sequence_len}")  # Debugging check
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

st.markdown("""
        ---
        <p style="text-align:center; font-size:20px; color:#00bfff;">
            Made by <a href="https://github.com/prince2004patel" style="color:#00bfff; text-decoration:none;"><b>Prince Patel</b></a>
        </p>
    """, unsafe_allow_html=True)