# Suicide Detection using Text Mining and Machine Learning

## Abstract

Suicide is a significant public health concern, and early identification of individuals at risk of suicidal behavior is crucial. However, it's a challenging task due to the complex nature of this behavior. This project leverages text mining and machine learning techniques to develop a high-precision model for identifying texts that may indicate suicidal behavior. The data used is sourced from Reddit, offering valuable insights into language patterns associated with risk.

## Contents

- [Methods](#methods)
  - [Data Preprocessing](#data-preprocessing)
  - [Vectorizations](#vectorizations)
  - [Feature Selection](#feature-selection)
  - [Plots](#plots)
- [Models and Results](#models-and-results)
- [Libraries Used](#libraries-used)

## Methods

### Data Preprocessing

- Text Pre-processing: Lowercasing, punctuation removal, stopword removal, and lemmatization were applied to clean the text data.

### Vectorizations

Vectorization techniques included Count Vectorization, TF-IDF Vectorization, and Bag of Words (BoW) Vectorization. Sentiment analysis using VADER was also performed.

### Feature Selection

Feature selection was achieved using LinearSVC to enhance model efficiency by focusing on the most relevant features while discarding less informative ones.

### Plots

Plots included visualizations of model performance metrics, ROC and precision-recall curves for binary classification models, and confusion matrices

## Models and Results

A combination of machine learning and deep learning methods was employed, including:

- Support Vector Machines (SVM)
- Random Forests
- Neural Networks
- Linear Regression
- Logistic Regression
- Naive Bayes

The best accuracy achieved for each model is as follows:

- Linear Regression: 92.5%
- Logistic Regression: 93.6%
- SVM: 93.5%
- Random Forest: 85.9%
- Naive Bayes: 90.8%
- Neural Networks: 93.1%

## Libraries Used

- pandas
- numpy
- seaborn
- matplotlib.pyplot
- plotly.express
- nltk
- gensim
- scikit-learn
- vaderSentiment

Feel free to explore the code and contribute to ongoing research in this critical area.
