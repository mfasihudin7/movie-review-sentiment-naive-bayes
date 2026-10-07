# 🎬 Movie Review Sentiment Classification using Naive Bayes

A simple **movie review sentiment classifier** built from scratch in Python using the **Multinomial Naive Bayes** algorithm.

The program reads movie reviews, processes their text, learns the relationship between words and sentiment, and then predicts whether unseen reviews are **positive** or **negative**.

> 📚 Developed as a Programming for AI project at FAST-NUCES.

---

## 📌 What is this project?

Have you ever written a movie review like:

> "The movie was amazing and I really enjoyed it."

A human can easily understand that this review is positive.

But how can a computer understand it?

This project demonstrates one way to solve this problem using **Natural Language Processing (NLP)** and **Naive Bayes classification**.

The program learns from a collection of reviews that are already labeled as:

- **Positive** 😊
- **Negative** 😞

It then uses what it learned to classify new, unseen reviews.

---

## 🧠 How Does It Work?

The project follows these main steps:

```text
Movie Reviews
      │
      ▼
Read the Reviews
      │
      ▼
Preprocess the Text
      │
      ▼
Remove Stop Words
      │
      ▼
Build Word Frequencies
      │
      ▼
Train Naive Bayes Model
      │
      ▼
Read Test Reviews
      │
      ▼
Calculate Positive & Negative Scores
      │
      ▼
Predict Sentiment
      │
      ▼
predictions.txt
