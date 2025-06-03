Sentiment Analysis using Machine Learning

This project focuses on building a sentiment analysis system that classifies text data as **positive** or **negative**. It demonstrates the complete machine learning workflow—from text preprocessing and vectorization to model training and evaluation—on a labeled dataset.

Project Overview

Objective:
To develop a binary classification model that can automatically determine the **sentiment** (positive or negative) of given text entries.

Dataset:
- File: `sentimentdataset.csv`
- Contains text data and their associated sentiment labels.
- Pre-labeled samples used for supervised learning.
- Structure:
  - `text`: The input sentence or review
  - `label`: Sentiment (`1` = positive, `0` = negative)

Tools and Technologies

- **Language**: Python
- **IDE**: Jupyter Notebook
- **Libraries Used**:
  - `pandas`, `numpy`
  - `sklearn` (for model training and evaluation)
  - `nltk` (for text preprocessing)
  - `matplotlib`, `seaborn` (for visualizations)

Project Workflow

 1. **Data Loading**
- Loaded the dataset using `pandas`.
- Checked for missing values and basic structure of the data.

 2. **Text Preprocessing**
- Converted text to lowercase
- Removed punctuation and stopwords using `nltk`
- Tokenization and optional stemming

 3. **Feature Engineering**
- Transformed text into numerical format using **TF-IDF Vectorizer**
- Split dataset into **training** and **test** sets

 4. **Model Training**
- Trained multiple classification models:
  - **Logistic Regression**
  - **Naive Bayes**
  - **Support Vector Machine (SVM)**

 5. **Model Evaluation**
- Evaluated performance using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

 6. **Visualization**
- Plotted confusion matrix and feature importance (where applicable)

 Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~89%     |
| Naive Bayes         | ~85%     |
| SVM                 | ~90%     |

SVM performed the best on this dataset, showing strong capability in classifying sentiment accurately.

 Sample Output

```python
Input: "The movie was absolutely wonderful and engaging!"
Predicted Sentiment: Positive

Input: "The plot was weak and the acting was worse."
Predicted Sentiment: Negative
