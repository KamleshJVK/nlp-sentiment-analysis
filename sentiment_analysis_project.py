
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sample data
data = {
    'review': [
        'I love this product! It works great.',
        'Terrible experience, will not buy again.',
        'Decent quality, but could be better.',
        'Excellent! Exceeded my expectations.',
        'Not worth the price.'
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['review'] = df['review'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['review']).toarray()
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Prediction on new data
new_reviews = [
    'This is the best purchase I have ever made!',
    'Worst product ever. Do not buy!'
]
new_reviews = [preprocess_text(review) for review in new_reviews]
new_X = vectorizer.transform(new_reviews).toarray()
predictions = model.predict(new_X)

print(f'Predictions: {predictions}')
