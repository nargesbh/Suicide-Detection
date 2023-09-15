# Suicide Detection using Text Mining and Machine Learning
![NLP gif](NLP.gif)

## Contents

- [Abstract](#abstract)
- [Methods](#methods)
  - [Data Preprocessing](#data-preprocessing)
  - [Vectorizations](#vectorizations)
  - [Feature Selection](#feature-selection)
  - [Plots](#plots)
- [Models and Results](#models-and-results)
- [Libraries Used](#libraries-used)

## Abstract

Suicide is a pressing public health concern with far-reaching implications for individuals and society as a whole. Early identification of individuals at risk of suicidal behavior is a critical step in preventing suicide. However, identifying individuals at risk of suicide can be a daunting challenge. Suicidal behavior is often a complex and multifaceted phenomenon that is deeply rooted in an individual's mental state and emotional well-being. Traditional methods of risk assessment may not always be sufficient, and this is where data-driven approaches can play a pivotal role.

This project explores the application of text mining and machine learning techniques to develop a high-precision model capable of identifying texts that may indicate suicidal behavior. The motivation behind this endeavor is the understanding that individuals in distress often seek an outlet to express their thoughts and emotions, and they may do so through text-based platforms, such as social media.

To achieve this objective, the project leverages text data obtained from the social media platform Reddit. Reddit serves as an abundant source of text data where individuals share their thoughts, feelings, and experiences. Analyzing this data provides a unique opportunity to uncover linguistic patterns and textual markers that could be indicative of suicidal ideation and behavior.

## Methods

### Data Preprocessing

Effective data preprocessing is a crucial step in preparing the text data for analysis. The following preprocessing steps were applied:

- **Text Pre-processing:** Text data was converted to lowercase to ensure uniformity in text representation. Furthermore, punctuation was removed to eliminate non-essential characters, making the text analysis-ready.

- **Stopword Removal:** Common stopwords, such as "the," "and," and "in," were removed from the text. This step aids in focusing on content-carrying words that are more likely to be indicative of suicidal behavior.

- **Lemmatization:** Lemmatization was employed to reduce words to their base or root forms. To achieve this, part-of-speech tagging using the Natural Language Toolkit (NLTK) was performed. Words were then lemmatized based on their respective parts of speech.

### Vectorizations

Text data, inherently unstructured, needed to be transformed into a numerical format for machine learning models. The following vectorization techniques were applied:

- **Count Vectorization (CountVectorizer):** Count Vectorization represents text data as a matrix of word counts. It quantifies the frequency of each word in a document.

- **TF-IDF Vectorization (TfidfVectorizer):** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization transforms text data into numerical values. It emphasizes words that are important within a specific document but less common across all documents. Additionally, a second TF-IDF Vectorizer (tfidf_vectorizer_n12) was employed, including bigrams (word pairs) to capture additional context.

- **Bag of Words (BoW) Vectorization (CountVectorizer):** BoW Vectorization represents text data as a collection of word frequencies. It treats each document as a vector of word occurrences, regardless of word order.

- **Sentiment Analysis with VADER:** Sentiment analysis was performed using the VADER sentiment analysis tool. A dedicated function, "get_vader_scores," was implemented to calculate sentiment scores, including negative, neutral, positive, and compound scores, for each document. These sentiment analysis results were added to the data as additional columns ('neg_score,' 'neu_score,' 'pos_score,' 'compound').

### Feature Selection

Feature selection is crucial for enhancing model efficiency and interpretability by focusing on the most relevant features while discarding less informative ones. Linear Support Vector Classifier (LinearSVC) was employed for feature selection. LinearSVC was chosen due to its L1 norm penalty, which encourages feature sparsity by assigning zero weights to less important features. This technique helps identify and retain the most influential features in the text data while reducing dimensionality.

### Plots

Visualizing and interpreting model performance is essential for gaining insights from the analysis. The project utilized several plots to convey key information:

- **plot_results(data) Function:** This function visualizes model performance metrics, including precision, recall, and F1-score, for different text representations. It employs bar charts with distinct colors for each representation.

- **plot_curves(model, X, y) Function:** This function generates ROC and precision-recall curves for binary classification models. The ROC curve assesses the trade-off between true positive and false positive rates, while the precision-recall curve evaluates the trade-off between precision and recall.

- **plot_confusion_matrices(cf_matrices, model_names) Function:** This function creates a grid of subplots, each containing a confusion matrix. Confusion matrices are vital for assessing model classification performance in multi-class scenarios.

## Models and Results

To achieve the goal of identifying texts indicating suicidal behavior, a combination of machine learning and deep learning models were employed:

- **Support Vector Machines (SVM)**
- **Random Forests**
- **Neural Networks**
- **Linear Regression**
- **Logistic Regression**
- **Naive Bayes**

The project aimed to identify the best-performing model based on accuracy. The following results were obtained:

- **Linear Regression:** Achieved an accuracy of 92.5%.
- **Logistic Regression:** Achieved an accuracy of 93.6%.
- **SVM:** Achieved an accuracy of 93.5%.
- **Random Forest:** Achieved an accuracy of 85.9%.
- **Naive Bayes:** Achieved an accuracy of 90.8%.
- **Neural Networks:** Achieved an accuracy of 93.1%.

## Libraries Used

The project made use of various Python libraries and tools to conduct data analysis and build machine learning models. Some of the libraries employed include:

- pandas
- numpy
- seaborn
- matplotlib.pyplot
- plotly.express
- nltk
- gensim
- scikit-learn
- vaderSentiment

This comprehensive analysis and model development process demonstrate the potential of text mining and machine learning in identifying individuals at risk of suicide. The project serves as a foundation for further research in this critical field and offers solutions to enhance public health by enabling early identification and intervention for individuals at risk of suicidal behavior. Contributors and researchers are encouraged to explore the code and continue advancing knowledge in this important area.
