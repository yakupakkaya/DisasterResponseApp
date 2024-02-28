# Disaster Response Pipeline Project

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. This project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweets and messages from real-life disasters. The project aims to build a Natural Language Processing (NLP) model to categorize messages.

## Project Components

There are three components of this project:

### 1. ETL Pipeline

A Python script, `process_data.py`, writes a data cleaning pipeline that:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 2. ML Pipeline

A Python script, `train_classifier.py`, writes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 3. Flask Web App

A Flask web app that visualizes the results of the models' classifications and allows the user to input new messages and get classification results in real-time.

## Getting Started

### Prerequisites

- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

### Instructions:

1. **Run the following commands in the project's root directory to set up your database and model.**

    - To run the ETL pipeline that cleans data and stores it in the database:
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run the ML pipeline that trains the classifier and saves it:
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. **Go to the `app` directory:**
    ```
    cd app
    ```

3. **Run your web app:**
    ```
    python run.py
    ```

4. **Follow the link provided by Flask to view the web app.**


## Acknowledgements

- Figure Eight for providing the relevant dataset to train the model.
- Udacity for offering guidance and project specifications.
