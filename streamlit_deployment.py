import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('word_predictor_lstm.h5') 
maxlen=model.input_shape[1]

with open ('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)


def predict_next_word(text):

    # Convert input text into word IDs
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Keep only the last 13 tokens if input is longer
    token_list = token_list[-(maxlen - 1):]

    # Add padding to make sequence length 13
    token_list = pad_sequences(
        [token_list],
        maxlen=maxlen - 1,
        padding='pre'
    )

    # Predict the next word
    predicted = model.predict(token_list, verbose=0)

    # Get the word ID with the highest probability
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Convert predicted ID back into a word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


st.title('Next Word Predictor')
sentence=st.text_input('Enter the sequence of words','i am a')

if st.button('Next word is'):
    next_word=predict_next_word(sentence)
    st.write('Your next word is ==>',next_word)
