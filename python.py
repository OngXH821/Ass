import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Function to fetch reviews from a URL (example only, replace with actual URL and parsing logic)
def fetch_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []
    for review in soup.find_all('div', class_='review'):
        text = review.get_text()
        reviews.append(text)
    return reviews

# Sample data (replace with actual data source)
data = {
    'review': [
        "This product is amazing! I love it.",
        "Worst purchase ever. Totally disappointed.",
        "The quality is okay, but it could be better.",
        "Fantastic! Highly recommend this.",
        "Not worth the money. I regret buying it."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['processed_review'] = df['review'].apply(preprocess_text)

# Vectorization and train-test split
X = df['processed_review']
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    print(f"{name} Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"{name} Classification Report:\n{classification_report(y_test, predictions)}")
