# PHASE 2: Data Loading & Preprocessing
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

# Download NLTK stopwords

nltk.download('stopwords')

#Load Dataset
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

#Add Labels
fake_df['label'] = 0   # 0 = Fake
true_df['label'] = 1   # 1 = Real

#Combine Both Datasets
df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"Total articles: {len(df)}")
print(f"Fake: {len(fake_df)} | Real: {len(true_df)}")
print(df.head())


#Keep Only Needed Columns
df = df[['text', 'label']]

#Drop empty rows
df.dropna(inplace=True)

#Text Cleaning Function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


    #Apply Cleaning
    print("\nCleaning text... (takes a minute)")
df['clean_text'] = df['text'].apply(clean_text)

print("\nCleaning Done! ✅")
print(df[['text', 'clean_text', 'label']].head(3))

#Save Cleaned Data
df.to_csv('data/cleaned_news.csv', index=False)
print("\nCleaned dataset saved → data/cleaned_news.csv ✅")


#-------------------------------------------------------------------------------------#

# PHASE 3: TF-IDF + Model Training + Evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


#TF-IDF Vectorization
print("\nApplying TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']
print(f"TF-IDF Shape: {X.shape} ✅")

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


#Train Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("── Naive Bayes Results ──")
print(f"Accuracy  : {accuracy_score(y_test, nb_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, nb_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, nb_pred)*100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, nb_pred)*100:.2f}%")

#Train Logistic Regression 
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("── Logistic Regression Results ──")
print(f"Accuracy  : {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"Precision : {precision_score(y_test, lr_pred)*100:.2f}%")
print(f"Recall    : {recall_score(y_test, lr_pred)*100:.2f}%")
print(f"F1-Score  : {f1_score(y_test, lr_pred)*100:.2f}%")

#Save Models
print("\nSaving models...")
pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
pickle.dump(lr_model, open('lr_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print("\n✅ All models saved!")
print("nb_model.pkl ✅")
print("lr_model.pkl ✅")
print("tfidf.pkl    ✅")
