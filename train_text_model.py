import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


#Loading data and converting to DataFrame
data_dir = "data"
emails = []

for label in ["ham", "spam"]:
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
        emails.append({"text": text, "label": label})
            
df = pd.DataFrame(emails)
print(df.head())
print(df['label'].value_counts())

#Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

#Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train_vec, y_train)

#Make predictions and evaluate the model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Save the trained model and vectorizer
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
print("Model and vectorizer saved to 'models/' directory.")
