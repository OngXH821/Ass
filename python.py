import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

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

# Predict with confidence scores
def predict(model, vectorizer, input_text):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    confidence = model.predict_proba(input_tfidf)[0]
    return prediction[0], confidence

# Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
    return accuracy, report

# Save model
def save_model(model, vectorizer):
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)

# Load model
def load_model():
    with open('model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

# Streamlit UI
st.title('Fake News Detection System')

# Load data
df = load_data()

# Display data sample
if st.checkbox('Show data sample'):
    st.write(df.head())

# Preprocess data and train model
X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
model = train_model(X_train_tfidf, y_train)

# Model evaluation
if st.checkbox('Evaluate model'):
    accuracy, report = evaluate_model(model, X_test_tfidf, y_test)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.text('Classification Report:')
    st.text(report)

# Get user input
user_input = st.text_area("Enter a sentence to check if it's Real or Fake news:", "")

if user_input:
    prediction, confidence = predict(model, vectorizer, user_input)
    if prediction == 'real':
        st.success(f"The news is likely Real! (Confidence: {confidence[1]:.2f})")
    else:
        st.error(f"The news is likely Fake! (Confidence: {confidence[0]:.2f})")

# Save model
if st.button('Save Model'):
    save_model(model, vectorizer)
    st.success('Model saved successfully!')

# Option to load saved model
if st.button('Load Model'):
    model, vectorizer = load_model()
    st.success('Model loaded successfully!')
