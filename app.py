import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    tokens = [[token.lower() for token in sentence] for sentence in tokens]
    tokens = [[token for token in sentence if token not in string.punctuation] for sentence in tokens]
    tokens = [[token for token in sentence if token not in stopwords.words('english')] for sentence in tokens]
    return [" ".join(sentence) for sentence in tokens], sentences

pdf_path = 'report.pdf'  
pdf_text = extract_text_from_pdf(pdf_path)
processed_text, original_sentences = preprocess_text(pdf_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_text)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = retrieve_information(user_input)
    return jsonify({"response": response})

def retrieve_information(query):
    query_vec = vectorizer.transform([query.lower()])
    results = (X * query_vec.T).toarray()
    relevant_indices = np.argsort(results.flatten())[::-1]
    top_n = 3  # Number of top results to return
    relevant_sentences = [original_sentences[i] for i in relevant_indices[:top_n]]
    return " ".join(relevant_sentences)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
