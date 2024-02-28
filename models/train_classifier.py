# import libraries
import os
import platform
import sys
import re
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import certifi

def configure_nltk_ssl_cert():
    """
    Configures SSL certificate for NLTK data download on macOS.

    This function sets the SSL_CERT_FILE environment variable to use certifi's certificate.
    It's a workaround for SSL certificate verification issues encountered on macOS.
    """
    if platform.system() == 'Darwin':  # Darwin is the system name for macOS
        os.environ['SSL_CERT_FILE'] = certifi.where()

def load_data(database_filepath):
    """
    Loads data from a SQLite database and returns feature and target variables.

    Args:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    X (DataFrame): Features dataset.
    Y (DataFrame): Target dataset.
    category_names (List[str]): List of category names.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('DisasterMessages', engine)
        X = df['message']
        Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
        category_names = Y.columns.tolist()
        return X, Y, category_names
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None, None


def tokenize(text):
    """
    Tokenizes text data.

    Args:
    text (str): Text to be tokenized.

    Returns:
    clean_tokens (List[str]): List of cleaned and lemmatized tokens.
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize, remove stop words and duplicate tokens
    stop_words = set(stopwords.words("english"))
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in stop_words]

    return list(set(clean_tokens))

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def build_model():
    """
    Builds a machine learning pipeline with GridSearchCV for parameter optimization.

    Returns:
    cv (GridSearchCV object): Configured GridSearchCV object with pipeline and parameter grid.
    """
    # Define the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameter grid for GridSearchCV
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model on the test dataset.

    Args:
    model: The trained model to evaluate.
    X_test: Test features.
    Y_test: Test labels.
    category_names: List of category names for classification.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Saves the trained model as a pickle file.

    Args:
    model: The trained model to save.
    model_filepath: Filepath for the saved model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to run the ML pipeline: load data, build, train, evaluate, and save the model.
    """
    if len(sys.argv) == 3:
        # Configure SSL certificate for NLTK on macOS to avoid download issues
        configure_nltk_ssl_cert()

        # Download necessary NLTK data
        nltk.download(["punkt", "wordnet", "stopwords"])

        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")
    else:
        print("Please provide the filepath of the disaster messages database "\
              "as the first argument and the filepath of the pickle file to "\
              "save the model to as the second argument. \n\nExample: python "\
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")

if __name__ == "__main__":
    main()
