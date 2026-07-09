import pickle
import streamlit as st
import nltk
import string

# from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('./vectorizer.pkl', 'rb'))
model = pickle.load(open('./model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
input_msg = st.text_area('Enter the Message')

if st.button('Predict'):

    # Preprocessing
    transform_msg = transform_text(input_msg)

    # Vectorize
    vector_input = tfidf.transform([transform_msg])

    # Predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')