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
    prediction_proba = model.predict_proba(input_tfidf)
    return prediction[0], prediction_proba

# Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    return report

# Streamlit UI
st.title('Fake News Detection System')

# Load data
df = load_data()

# Preprocess data and train model
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
model = train_model(X_train_tfidf, y_train)

# Display model evaluation report
if st.checkbox("Show Model Evaluation"):
    st.subheader("Model Evaluation on Test Data")
    report = evaluate_model(model, X_test_tfidf, y_test)
    st.text(report)

# Get user input
user_input = st.text_area("Enter a sentence to check if it's Real or Fake news:", "")

if user_input:
    prediction, prediction_proba = predict(model, vectorizer, user_input)
    proba_real = prediction_proba[0][0]
    proba_fake = prediction_proba[0][1]
    
    if prediction == 'real':
        st.success(f"The news is likely Real with a probability of {proba_real:.2f}!")
    else:
        st.error(f"The news is likely Fake with a probability of {proba_fake:.2f}!")
    
    # Display probabilities
    st.write(f"Real News Probability: {proba_real:.2f}")
    st.write(f"Fake News Probability: {proba_fake:.2f}")
