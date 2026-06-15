# =============================================
# PHASE 4: Flask Web Application (Updated V2)
# =============================================

from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# ── Load V1 Models (Political News) ──────────
tfidf_v1    = pickle.load(open('tfidf.pkl',        'rb'))
nb_model_v1 = pickle.load(open('nb_model.pkl',     'rb'))
lr_model_v1 = pickle.load(open('lr_model.pkl',     'rb'))

# ── Load V2 Models (Global/Multi-Domain) ─────
tfidf_v2    = pickle.load(open('tfidf_v2.pkl',     'rb'))
nb_model_v2 = pickle.load(open('nb_model_v2.pkl',  'rb'))
lr_model_v2 = pickle.load(open('lr_model_v2.pkl',  'rb'))

stop_words = set(stopwords.words('english'))

# ── Text Cleaning Function ────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ── Routes ────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text  = request.form['news_text']
    model_choice = request.form.get('model_choice', 'v2')  # Default V2

    # Clean Text
    cleaned = clean_text(news_text)

    # Model Selection
    if model_choice == 'v1':
        vectorized = tfidf_v1.transform([cleaned])
        nb_result  = nb_model_v1.predict(vectorized)[0]
        lr_result  = lr_model_v1.predict(vectorized)[0]
        nb_prob    = nb_model_v1.predict_proba(vectorized)[0]
        lr_prob    = lr_model_v1.predict_proba(vectorized)[0]
        model_label = "V1 — Political Dataset (44,898 articles)"
    else:
        vectorized = tfidf_v2.transform([cleaned])
        nb_result  = nb_model_v2.predict(vectorized)[0]
        lr_result  = lr_model_v2.predict(vectorized)[0]
        nb_prob    = nb_model_v2.predict_proba(vectorized)[0]
        lr_prob    = lr_model_v2.predict_proba(vectorized)[0]
        model_label = "V2 — Global Dataset (116,993 articles)"

    # Final Decision
    final_result = "REAL ✅" if lr_result == 1 else "FAKE ❌"
    confidence   = round(max(lr_prob) * 100, 2)

    return render_template('index.html',
        prediction    = final_result,
        confidence    = confidence,
        nb_result     = "Real ✅" if nb_result == 1 else "Fake ❌",
        lr_result     = "Real ✅" if lr_result == 1 else "Fake ❌",
        nb_confidence = round(max(nb_prob) * 100, 2),
        lr_confidence = round(max(lr_prob) * 100, 2),
        news_text     = news_text,
        model_choice  = model_choice,
        model_label   = model_label
    )

if __name__ == '__main__':
    app.run(debug=True)