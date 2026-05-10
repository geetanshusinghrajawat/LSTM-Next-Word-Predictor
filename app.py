import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load Model
model=load_model('nextword_model.h5')
with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)

reverse_index={idx:word for word,idx in tokenizer.word_index.items()}
max_len=185

def generate_text(seed_text,num_words=10):
    text=seed_text
    for _ in range(num_words):
        seq=tokenizer.texts_to_sequences([text])[0]
        padded=pad_sequences([seq],maxlen=max_len,padding='pre')
        preds=model.predict(padded,verbose=0)
        pos=np.argmax(preds)
        next_word=reverse_index.get(pos," ")
        text+=" "+next_word
    return text

st.title("Next Word Prediction Using LSTM")

seed = st.text_input('Enter a Starting Text: ','Hello')
num_words=st.slider('Number of words to predict',1,5,3)

if st.button("Generate Predicted Words"):
    result=generate_text(seed,num_words)
    st.write(result)