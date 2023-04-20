import streamlit as st
import pickle
from utils import transform_text

def app():    

    tfidf = pickle.load((open('models/vectorizer.pkl','rb')))
    model = pickle.load((open('models/model.pkl','rb')))
    st.title("Email Spam Classifier")
    input_sms = st.text_area("Enter the message")
    if st.button('Predict'):

        #1. preprocess
        transformed_sms = transform_text(input_sms)
        #2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        #3. predict
        result = model.predict(vector_input)[0]
        #4. Display
        if result == 1:
            st.warning("Spam", icon="❌")
        else:
            st.success("Not Spam", icon="✅")