# 📰 TruthLens – Fake News Detection System

TruthLens is an AI-powered Fake News Detection System developed using Machine Learning and Natural Language Processing (NLP). The system analyzes news articles and predicts whether the news is **REAL** or **FAKE** with confidence scores.

## 🚀 Features

- Fake News Detection using Machine Learning
- NLP-based Text Preprocessing
- TF-IDF Feature Extraction
- Dual Model Prediction
  - Naive Bayes
  - Logistic Regression
- Confidence Score Calculation
- Flask Web Application
- Support for Political and Multi-Domain News
- Real-Time Prediction Interface

---

## 📊 Datasets Used

### Model V1 (Political News)

- Fake.csv – 23,481 Articles
- True.csv – 21,417 Articles

**Total:** 44,898 Articles

### Model V2 (Multi-Domain News)

- Fake.csv
- True.csv
- WELFake Dataset – 72,134 Articles

**Total:** 116,993 Articles

Domains Covered:

- Politics
- Sports
- Technology
- Entertainment

---

## 🔄 Data Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Remove special characters
- Remove numbers
- Remove extra spaces
- Remove stopwords using NLTK

---

## 🧠 Machine Learning Models

### Naive Bayes

- MultinomialNB Classifier
- Fast text classification
- Used for comparison and reference prediction

### Logistic Regression

- Primary prediction model
- Higher accuracy
- Generates final verdict

---

## 📈 Model Performance

| Model | Accuracy |
|---------|---------|
| Naive Bayes V1 | 93.54% |
| Logistic Regression V1 | 98.91% |
| Naive Bayes V2 | 87.13% |
| Logistic Regression V2 | 96.53% |

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- NLTK
- Pandas
- NumPy
- TF-IDF
- Pickle
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```bash
TruthLens/
│
├── app.py
├── model.py
├── model_v2.py
│
├── tfidf.pkl
├── nb_model.pkl
├── lr_model.pkl
│
├── tfidf_v2.pkl
├── nb_model_v2.pkl
├── lr_model_v2.pkl
│
├── requirements.txt
├── templates/
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/TruthLens.git
cd TruthLens
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

## 💡 How It Works

1. User enters a news article.
2. Text is cleaned using NLP preprocessing.
3. TF-IDF converts text into numerical vectors.
4. Naive Bayes and Logistic Regression generate predictions.
5. Logistic Regression provides the final verdict.
6. Confidence score is displayed to the user.

---

## 🎯 Project Objective

The objective of TruthLens is to combat misinformation by providing an automated solution for fake news detection using Artificial Intelligence, Natural Language Processing, and Machine Learning techniques.

---

## 👨‍💻 Team Members

**Parthum Kumar**  
Machine Learning Engineer

**Rohit Talreja**  
UI Developer

---

## 🎓 Academic Information

**Course:** Artificial Intelligence

**Institution:** Iqra University

**Instructor:** Miss Noureen Fatima

---

## 📜 License

This project is developed for educational and academic purposes only.
