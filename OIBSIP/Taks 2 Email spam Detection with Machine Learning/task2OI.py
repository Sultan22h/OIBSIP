# Task 2: Email Spam Detection with Machine Learning By Sultan Hussain

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1. Load Data
# -------------------------------
# Force only first two columns, no extra unnamed ones
data = pd.read_excel(
    r"C:\Users\DELL\Desktop\OIBSIP\Taks 2 Email spam Detection with Machine Learning\spam.xlsx",
    usecols=[0, 1],   # take only v1 and v2
    dtype=str         # make sure everything is read as string
)

print("Data Preview:\n", data.head())

# Rename columns
data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# -------------------------------
# 2. Data Cleaning
# -------------------------------
# Drop missing rows
data.dropna(inplace=True)

# Encode labels: ham=0, spam=1
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# 3. Exploratory Data Analysis
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='label')
plt.title("Distribution of Spam vs Ham Emails")
plt.show()

# -------------------------------
# 4. Text Processing & Train/Test Split
# -------------------------------
X = data['message']   # features (email text)
y = data['label_num'] # target (0 or 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into numerical features (Bag of Words)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# -------------------------------
# 5. Model Training (Naive Bayes)
# -------------------------------
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# -------------------------------
# 6. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test_counts)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# 7. Test with Custom Input
# -------------------------------
def predict_email(email):
    email_counts = vectorizer.transform([email])
    prediction = model.predict(email_counts)[0]
    return "Spam" if prediction == 1 else "Ham (Not Spam)"

# Example
print("\nTest Prediction:")
print("Message: 'You have won $1000 cash prize!!!'")
print("Prediction:", predict_email("You have won $1000 cash prize!!!"))

print("Message: 'Hi, are we still meeting tomorrow?'")
print("Prediction:", predict_email("Hi, are we still meeting tomorrow?"))
