Email Spam Classifier Using Machine Learning

Author: Shahzaib Asif
Track: Machine Learning Internship (Arch Technologies)
Submission Date: Before 27th of month

Project Overview

This project is a Machine Learning–based Email Spam Classifier that predicts whether an email is Spam or Not Spam using:

TF-IDF Vectorization

Support Vector Machine (SVM) Classifier

Custom Text Preprocessing

Interactive Streamlit Web App

The frontend Streamlit app allows users to paste any email text and receive instant predictions.

Files Included
File	Description
spam.ipynb	Training notebook (data cleaning → training → evaluation → model saving)
app.py	Streamlit frontend (loads model + vectorizer and predicts spam/ham)
svm_spam_model.pkl	Trained SVM model
tfidf_vectorizer.pkl	TF-IDF vectorizer used during training
dataset.csv	Spam/Ham dataset (if included)
README.md	Project documentation
requirements.txt	List of required Python libraries
How It Works

User enters/pastes an email in the Streamlit UI

Text is cleaned (lowercase, remove URLs, punctuation, stopwords, stemming)

Cleaned text is converted to TF-IDF vector

Vector is passed to the trained SVM model

Output:

Spam (1)

Not Spam (0)

Model Performance

The SVM classifier gives:

High Accuracy

High Precision

High Recall

Excellent performance on unseen emails

(Exact scores are shown inside spam.ipynb.)

How to Run the App
1. Install Requirements
pip install -r requirements.txt

2. Run Streamlit App
streamlit run app.py

Libraries Used

scikit-learn

pandas

numpy

nltk

streamlit

matplotlib, seaborn (for EDA)

Author

Shahzaib Asif
BSIT – 7th Semester
Machine Learning Enthusiast