# Movie Review Sentiment Classification using Naive Bayes

A Python implementation of a movie review sentiment classifier using the **Multinomial Naive Bayes** algorithm.

The project reads positive and negative movie reviews, preprocesses the text, builds word-frequency statistics, calculates class probabilities using Laplace smoothing, and classifies previously unseen reviews as either **POSITIVE** or **NEGATIVE**.

## Project Overview

This project was developed for the **Programming for AI** course at FAST-NUCES.

The classifier is implemented from scratch using Python without using ready-made machine learning classifiers such as `MultinomialNB`.

## Features

* Reads positive, negative, and test review files
* Converts review text to lowercase
* Removes punctuation and symbols
* Tokenizes reviews into words
* Removes English stop words
* Builds separate word-frequency dictionaries for positive and negative reviews
* Calculates class prior probabilities
* Uses Laplace smoothing
* Uses logarithmic probabilities for Naive Bayes scoring
* Classifies test reviews as POSITIVE or NEGATIVE
* Generates a `predictions.txt` file containing the predictions

## Dataset

The provided dataset contains:

* 900 positive movie reviews
* 900 negative movie reviews
* 200 unlabeled test reviews
* An English stop-word list

The expected directory structure is:

```text
data/
├── english.stop
└── imdb1/
    ├── pos/
    ├── neg/
    └── test/
```

## How It Works

### 1. Read Reviews

The program loads the review files from the positive, negative, and test folders.

### 2. Text Preprocessing

Each review is:

1. Converted to lowercase
2. Cleaned by removing punctuation
3. Split into individual words
4. Processed to remove stop words

### 3. Build Vocabulary

The program creates word-frequency dictionaries for both the positive and negative classes.

The vocabulary size is calculated using the unique words appearing in the positive and negative training data.

### 4. Naive Bayes Classification

The classifier calculates prior probabilities for the positive and negative classes.

For each word, Laplace smoothing is applied:

```text
P(word | class) = (word_count + 1) / (total_words + vocabulary_size)
```

Log probabilities are used when calculating the final scores to avoid numerical underflow.

### 5. Prediction

For every test review, the positive and negative scores are compared.

The class with the higher score becomes the prediction:

```text
POSITIVE
```

or

```text
NEGATIVE
```

### 6. Prediction File

The program generates:

```text
predictions.txt
```

Each line contains the test filename and its predicted sentiment:

```text
filename.txt, POSITIVE
filename.txt, NEGATIVE
```

## Project Structure

```text
Movie-Review-Sentiment-Naive-Bayes/
│
├── main.py
├── predictions.txt
├── README.md
│
└── data/
    ├── english.stop
    │
    └── imdb1/
        ├── pos/
        ├── neg/
        └── test/
```

## Requirements

* Python 3
* No external machine learning libraries are required

The implementation uses Python's built-in:

```python
os
math
```

## How to Run

Make sure the project has the required `data` directory and its files.

Then run:

```bash
python main.py
```

The program will display the number of positive, negative, and test reviews, build the Naive Bayes model, classify the test reviews, and create:

```text
predictions.txt
```

## Course

**Programming for AI — FAST-NUCES**

### Topics Demonstrated

* Python file handling
* Strings
* Lists
* Dictionaries
* Sets
* Loops
* Conditional statements
* Functions
* Text preprocessing
* Word-frequency counting
* Probability
* Laplace smoothing
* Naive Bayes classification
* Log probabilities
* File generation
