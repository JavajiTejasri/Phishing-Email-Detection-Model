import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("emails.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.3,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Output
print("\nPhishing Email Detection Model")
print("=" * 40)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

# Test custom email
email = ["Click here to claim your free reward now"]

email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

print("\nTest Email Prediction:")
print(f"Email: {email[0]}")
print(f"Prediction: {prediction[0]}")