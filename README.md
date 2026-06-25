# AI Sentiment Analysis System Using Naive Bayes

# Project Overview

The AI Sentiment Analysis System is a Machine Learning and Natural Language Processing (NLP) project developed to automatically determine the sentiment expressed in textual feedback. The system analyzes user reviews, comments, and feedback and classifies them as **Positive**, **Negative**, or **Neutral** sentiment.

The project uses the **IMDb Movie Reviews Dataset** containing 50,000 labeled reviews and implements a complete NLP pipeline including text preprocessing, TF-IDF vectorization, sentiment classification, and a Flask-based web interface.

---

# Objectives

* Perform sentiment analysis on textual feedback.
* Apply Natural Language Processing (NLP) techniques for text cleaning.
* Convert text into numerical features using TF-IDF.
* Train a Machine Learning model using Multinomial Naive Bayes.
* Provide sentiment predictions with confidence scores.
* Build an interactive web application using Flask.

---

# Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Emoji Library
* HTML
* CSS
* Bootstrap 5

---

# Dataset

**Dataset Used:** IMDb Movie Reviews Dataset

* Total Reviews: 50,000
* Positive Reviews: 25,000
* Negative Reviews: 25,000

Dataset columns:

| Column    | Description                |
| --------- | -------------------------- |
| review    | Movie review text          |
| sentiment | Positive or Negative label |

---

# Methodology

## 1. Data Preprocessing

The text data is cleaned using the following steps:

* Convert text to lowercase
* Remove URLs
* Remove special characters
* Handle emojis
* Remove unnecessary symbols

## 2. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization is used to convert text into numerical vectors.

## 3. Model Training

The project uses:

**Multinomial Naive Bayes Classifier**

The dataset is divided into:

* Training Set: 80%
* Testing Set: 20%

## 4. Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

---

# Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.78% |
| Precision | 87%    |
| Recall    | 87%    |
| F1-Score  | 87%    |

---

# Features

* Sentiment Prediction
* Confidence Score Display
* Emoji Support
* Modern User Interface
* Flask Web Application
* Real-time Analysis
* NLP-based Text Processing

---

# Project Structure

```text
Sentiment_Analysis_Project
│
├── dataset
│   └── IMDB Dataset.csv
│
├── model
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── templates
│   └── index.html
│
├── static
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

# Installation

Clone the repository:

```bash
git clone https://github.com/arpita-mewada/Sentiment-Analysis-Using-Naive-Bayes.git
```

Move into the project folder:

```bash
cd Sentiment-Analysis-Using-Naive-Bayes
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the application:

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

# Sample Inputs

## Positive Review

```text
I love this movie ❤️
```

Output:

```text
Positive Sentiment
```

## Negative Review

```text
Worst movie ever 😡
```

Output:

```text
Negative Sentiment
```

## Neutral Review

```text
This movie was okay.
```

Output:

```text
Neutral Sentiment
```

---

# Future Enhancements

* Multilingual Sentiment Analysis
* Sarcasm Detection
* Deep Learning Models
* BERT-based Sentiment Analysis
* Social Media Sentiment Monitoring
* Real-time Analytics Dashboard

---

#  Author

**Arpita Mewada**

Integrated M.Tech in Artificial Intelligence

VIT Bhopal University

---

#  Project Outcome

Successfully developed an AI-powered Sentiment Analysis System capable of classifying textual feedback with an accuracy of **86.78%** using TF-IDF and Multinomial Naive Bayes, integrated with a Flask-based web interface for real-time sentiment prediction.
