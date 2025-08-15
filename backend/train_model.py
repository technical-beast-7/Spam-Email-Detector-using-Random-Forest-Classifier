import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report  # ✅ Add metrics
import pickle
import os

# Get the directory of the current script to build robust paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset', 'email_dataset_full_with_email.csv')
model_dir = os.path.join(script_dir, 'model')
model_path = os.path.join(model_dir, 'spam_classifier.pkl')

# Load dataset
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The dataset file was not found at {os.path.abspath(dataset_path)}. Please check the file path.")

# Check if required columns exist
required_columns = ['email_id', 'subject', 'body', 'label']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Combine text fields and map labels
df['text'] = df['email_id'].fillna('') + ' ' + df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

if df['label'].isnull().any():
    raise ValueError("The 'label' column contains invalid or unmapped values.")

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ Predict on test set
y_pred = model.predict(X_test)

# ✅ Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Print classification report
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
print("\n📋 Classification Report:\n")
print(report)

# Save model and vectorizer
os.makedirs(model_dir, exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model and vectorizer saved successfully!")
