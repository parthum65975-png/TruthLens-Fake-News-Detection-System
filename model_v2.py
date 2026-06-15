# MODEL V2 — Multi-Domain Global Dataset
# Original Political + WELFake Global News

import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords')

print("=" * 50)
print("  MODEL V2 — Multi-Domain Training")
print("=" * 50)

#Load Original Dataset
print("\n[1/6] Loading Original Political Dataset...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['label'] = 0
true_df['label'] = 1

df_original = pd.concat([
    fake_df[['text', 'label']],
    true_df[['text', 'label']]
], ignore_index=True)

print(f"      Original dataset: {len(df_original)} articles")

#Load WELFake Dataset
print("\n[2/6] Loading WELFake Global Dataset...")
wel_df = pd.read_csv('data/WELFake_Dataset.csv')
print(f"      WELFake columns: {list(wel_df.columns)}")

#check text column in WELFake
if 'text' in wel_df.columns and 'label' in wel_df.columns:
    wel_df = wel_df[['text', 'label']]
elif 'Text' in wel_df.columns:
    wel_df = wel_df.rename(columns={'Text': 'text', 'Label': 'label'})
    wel_df = wel_df[['text', 'label']]
    # WELFake labels flip karo — 0=Real→1, 1=Fake→0
wel_df['label'] = wel_df['label'].map({0: 1, 1: 0})

wel_df.dropna(inplace=True)
print(f"      WELFake dataset: {len(wel_df)} articles")
print(f"      Fake: {len(wel_df[wel_df['label']==0])} | Real: {len(wel_df[wel_df['label']==1])}")

#Combine Both
print("\n[3/6] Combining Datasets...")
df = pd.concat([df_original, wel_df], ignore_index=True)
df.dropna(inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

print(f"      Total articles: {len(df)}")
print(f"      Total Fake: {len(df[df['label']==0])}")
print(f"      Total Real: {len(df[df['label']==1])}")

#Clean Text
print("\n[4/6] Cleaning Text... (2-3 minutes)")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("      Cleaning Done! ✅")

#TF-IDF 
print("\n[5/6] Applying TF-IDF Vectorization...")
tfidf_v2 = TfidfVectorizer(max_features=10000)  # V1 se double features
X = tfidf_v2.fit_transform(df['clean_text'])
y = df['label']
print(f"      TF-IDF Shape: {X.shape} ✅")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

#Train Models
print("\n[6/6] Training Models...")

# Naive Bayes
print("\n      Training Naive Bayes V2...")
nb_v2 = MultinomialNB()
nb_v2.fit(X_train, y_train)
nb_pred = nb_v2.predict(X_test)

print("\n      ── Naive Bayes V2 Results ──")
print(f"      Accuracy  : {accuracy_score(y_test, nb_pred)*100:.2f}%")
print(f"      Precision : {precision_score(y_test, nb_pred)*100:.2f}%")
print(f"      Recall    : {recall_score(y_test, nb_pred)*100:.2f}%")
print(f"      F1-Score  : {f1_score(y_test, nb_pred)*100:.2f}%")

# Logistic Regression
print("\n      Training Logistic Regression V2...")
lr_v2 = LogisticRegression(max_iter=1000)
lr_v2.fit(X_train, y_train)
lr_pred = lr_v2.predict(X_test)

print("\n      ── Logistic Regression V2 Results ──")
print(f"      Accuracy  : {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"      Precision : {precision_score(y_test, lr_pred)*100:.2f}%")
print(f"      Recall    : {recall_score(y_test, lr_pred)*100:.2f}%")
print(f"      F1-Score  : {f1_score(y_test, lr_pred)*100:.2f}%")

#Compare V1 vs V2
print("\n" + "=" * 50)
print("  V1 vs V2 COMPARISON")
print("=" * 50)
print(f"  V1 Dataset    : 44,898 articles (Political only)")
print(f"  V2 Dataset    : {len(df)} articles (Political + Global)")
print(f"  V1 LR Accuracy: 98.91%")
print(f"  V2 LR Accuracy: {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"  V2 Coverage   : Politics + Entertainment + Sports + Tech")
print("=" * 50)

#Save V2 Models
print("\nSaving V2 Models...")
pickle.dump(nb_v2,    open('nb_model_v2.pkl',  'wb'))
pickle.dump(lr_v2,    open('lr_model_v2.pkl',  'wb'))
pickle.dump(tfidf_v2, open('tfidf_v2.pkl',     'wb'))

print("\n✅ V2 Models Saved!")
print("   nb_model_v2.pkl  ✅")
print("   lr_model_v2.pkl  ✅")
print("   tfidf_v2.pkl     ✅")