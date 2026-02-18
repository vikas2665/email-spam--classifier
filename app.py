import streamlit as st
import pickle
import re

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Email / SMS Spam Detection")

user_input = st.text_area("Enter message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if pred == 1:
            st.error(f"🚨 Spam (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {prob[0]*100:.2f}%)")
