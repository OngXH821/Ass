import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('english_test_with_labels.csv')  # Assume dataset has 'text' and 'sentiment' columns
    return df

# Preprocess data
def preprocess_data(df):
    X = df['text']  # Column containing the text data
    y = df['sentiment']  # Column containing the labels (positive, negative, neutral)
    
    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# Train models
def train_models(X_train_tfidf, y_train):
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    return lr_model, nb_model

# Predict
def predict(model, vectorizer, input_text):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    return prediction[0]

# Streamlit UI
st.title('Sentiment Analysis System')

# Load data
df = load_data()

# Preprocess data and train models
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
lr_model, nb_model = train_models(X_train_tfidf, y_train)

# Display options for model selection
model_choice = st.selectbox('Choose a model', ['Logistic Regression', 'Naive Bayes'])

# Get user input
user_input = st.text_area("Enter a sentence to classify its sentiment:", "")

if user_input:
    if model_choice == 'Logistic Regression':
        prediction = predict(lr_model, vectorizer, user_input)
    else:
        prediction = predict(nb_model, vectorizer, user_input)
    
    st.write(f'The sentiment is: {prediction}')

# Show classification report for model comparison
st.write("Model Comparison Report")
lr_report = classification_report(y_test, lr_model.predict(X_test_tfidf), output_dict=True)
nb_report = classification_report(y_test, nb_model.predict(X_test_tfidf), output_dict=True)

st.write(f"Logistic Regression:\n {classification_report(y_test, lr_model.predict(X_test_tfidf))}")
st.write(f"Naive Bayes:\n {classification_report(y_test, nb_model.predict(X_test_tfidf))}")
