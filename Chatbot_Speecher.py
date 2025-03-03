import nltk
import streamlit as st
import speech_recognition as sr
import re
import random
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('stopwords')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Load dataset
data = pd.read_csv('Samsung Dialog.txt', sep=':', header=None)
cust = data.loc[data[0] == 'Customer'][1].reset_index(drop=True)
sales = data.loc[data[0] == 'Sales Agent'][1].reset_index(drop=True)
new_data = pd.DataFrame({'Question': cust, 'Answers': sales})

# Preprocessing function
def preprocessing(text):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentences.append(' '.join(tokens))
    return ' '.join(preprocessed_sentences)

new_data['tokenized Questions'] = new_data['Question'].apply(preprocessing)
xtrain = new_data['tokenized Questions'].to_list()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

# Speech-to-Text Function
def transcribe_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.write("Listening... Please speak now.")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand your speech."
    except sr.RequestError:
        return "Sorry, the speech recognition service is unavailable."

# Chatbot Greetings & Farewell
bot_greeting = ["Hello! Do you have any questions?", "Hey there! Ask me anything.", "Hi! How can I help you today?"]
bot_farewell = ["Thanks for using this chatbot. Goodbye!", "Have a great day and enjoy Samsung!", "See you later!"]
human_greeting = ["hi", "hello", "good day", "hey", "hola"]
human_exit = ["thank you", "thanks", "bye", "goodbye", "quit"]

# Streamlit UI
st.markdown("<h1 style='color:rgb(60, 30, 230); text-align: center; font-size: 50px;'>Speech-Enabled Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:rgb(133, 200, 231);'>Built by KufreKing</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


st.header('Background Of Study', divider = True)
st.write('A chatbot is an AI-powered program designed to simulate human conversation. It can interact with users through text or voice, answering questions, providing assistance, or performing tasks. Chatbots are commonly used in customer service, marketing, and entertainment, leveraging natural language processing (NLP)to understand and respond to queries.')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col2.image('pngwing.com (3).png')

# User Input Section
userPrompt = st.chat_input("Ask Your Question")

# Speech Input Button
if st.button("🎤 Speak"):
    userPrompt = transcribe_speech()
    st.write(f"Recognized Speech: **{userPrompt}**")

if userPrompt:
    col1.chat_message("user").write(userPrompt)
    userPrompt = userPrompt.lower()

    if userPrompt in human_greeting:
        col1.chat_message("ai").write(random.choice(bot_greeting))
    elif userPrompt in human_exit:
        col1.chat_message("ai").write(random.choice(bot_farewell))
    else:
        preUserInput = preprocessing(userPrompt)
        vect_user = tfidf_vectorizer.transform([preUserInput])
        similarity_scores = cosine_similarity(vect_user, corpus)
        most_similarity_index = np.argmax(similarity_scores)
        col1.chat_message("ai").write(new_data['Answers'].iloc[most_similarity_index])
