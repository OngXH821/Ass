import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('english_test_with_labels.csv')
    return df

# Preprocess data
def preprocess_data(df):
    X = df['tweet']
    y = df['label']
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train the model
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model

# Predict
def predict(model, vectorizer, input_text):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    return prediction[0]

# Streamlit UI
st.title('Fake News Detection System')

# Load data
df = load_data()

# Preprocess data and train model
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
model = train_model(X_train_tfidf, y_train)

# Get user input
user_input = st.text_area("Enter a sentence to check if it's Real or Fake news:", "")

if user_input:
    prediction = predict(model, vectorizer, user_input)
    if prediction == 'real':
        st.success("The news is likely Real!")
    else:
        st.error("The news is likely Fake!")
