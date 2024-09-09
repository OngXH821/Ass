import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Web Crawler (Example)
def crawl_forum_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text from the forum (adjust based on the structure of the forum)
    posts = soup.find_all('div', class_='post-content')
    data = [post.text.strip() for post in posts]
    
    return pd.DataFrame(data, columns=['tweet'])  # Returning as a DataFrame for consistency


# 2. Load Dataset (You can replace it with your own dataset)
@st.cache_data
def load_data():
    # You can load a dataset from CSV or crawl data using `crawl_forum_data()`
    df = pd.read_csv('english_test_with_labels.csv')  # Use an existing dataset if available
    return df


# 3. Preprocessing the Data
def preprocess_data(df):
    X = df['tweet']  # Text data
    y = df['label']  # Labels (positive, negative, or neutral)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# 4. Train Different Models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Store the evaluation results
    results = []
    
    for name, model in classifiers.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Append results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    return pd.DataFrame(results)


# 5. Predict Single Input (from user)
def predict_input(model, vectorizer, input_text):
    input_tfidf = vectorizer.transform([input_text])
    prediction = model.predict(input_tfidf)
    return prediction[0]


# 6. Streamlit Web App Interface
def run_app():
    st.title('Sentiment Analysis System')

    # Load data (either crawl or load from a dataset)
    df = load_data()

    # Preprocess data and split into training and test sets
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
    
    # Train and evaluate models
    results_df = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # Display results
    st.write("### Model Evaluation Results")
    st.dataframe(results_df)
    
    # Allow user to enter text for sentiment analysis
    user_input = st.text_area("Enter a sentence to check its sentiment:")
    
    if user_input:
        # Default to using Logistic Regression for the prediction
        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)
        
        prediction = predict_input(model, vectorizer, user_input)
        st.write(f"The predicted sentiment is: {prediction}")

# Run the app
if __name__ == '__main__':
    run_app()
