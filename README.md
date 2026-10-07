# Movie Review Sentiment Classification using Naive Bayes

## Overview

This project implements a **Movie Review Sentiment Classification system** using a **Multinomial Naive Bayes classifier** from scratch in Python.

The program reads positive and negative movie reviews, preprocesses the text, builds a vocabulary, calculates probabilities using Laplace smoothing, and classifies unseen movie reviews as **POSITIVE** or **NEGATIVE**.

This project was developed for the **Programming for AI** course at **FAST-NUCES**.

## Features

- Reads positive, negative, and test reviews from folders
- Converts text to lowercase
- Removes punctuation
- Tokenizes reviews into words
- Removes stop words
- Builds positive and negative word-frequency dictionaries
- Calculates class priors
- Uses Laplace smoothing
- Uses logarithmic probabilities
- Classifies test reviews as POSITIVE or NEGATIVE
- Generates a `predictions.txt` file containing the final predictions

## Dataset

The dataset contains:

- 900 positive movie reviews
- 900 negative movie reviews
- 200 unlabeled test reviews
- An English stop-word file

### Dataset Structure

After extracting `data.zip`, the project should have the following structure:

```text
movie-review-sentiment-naive-bayes/
│
├── main.py
├── README.md
├── predictions.txt
├── .gitignore
├── data.zip
│
└── data/
    ├── english.stop
    │
    └── imdb1/
        ├── pos/
        ├── neg/
        └── test/
