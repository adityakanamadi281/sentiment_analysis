import streamlit as st
import pandas as pd
import numpy as np
import joblib

sentiment_model = joblib.load(r"C:\Users\adity\Sentiment Analysis\sentiment_model.joblib")
vectorizer = joblib.load(r"C:\Users\adity\Sentiment Analysis\tfidf_vectorizer.pkl")



st.title("Sentiment Analysis Prediction App")
st.write("Enter details below to predict sentiment.")

title = st.text_input("title of feedback", placeholder="Enter title here")
rating = st.number_input("Rating", min_value=1,max_value=5, value=3)
body = st.text_input("Body", placeholder="Enter the body of feedback here")


combined_text = title + " " + body

input_data = pd.DataFrame(
    {
        "title":[title],
        "body" : [body]
    }
)

input_vectorized = vectorizer.transform(input_data)

if st.button("Predict Sentiment "):
    sentiment = sentiment_model.predict(input_vectorized)[0]
    st.success(f"Predicted Sentiment : sentiment {sentiment}")