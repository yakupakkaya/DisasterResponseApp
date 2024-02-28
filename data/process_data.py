import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges messages and categories datasets from specified filepaths.

    Args:
    messages_filepath (str): The file path for the messages dataset.
    categories_filepath (str): The file path for the categories dataset.

    Returns:
    df (DataFrame): Merged DataFrame of messages and categories on 'id'.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on="id")

    return df

def clean_data(df):
    """
    Cleans the merged message and categories data frame. Converts the category values to binary.
    If category values are higher than "1", they are assumed as category "1".

    Args:
    df (DataFrame): The merged data frame of messages and categories.

    Returns:
    DataFrame: The cleaned data frame with categories expanded and duplicates removed.
    """
    # Expand the categories column into separate, clearly named columns
    categories = df['categories'].str.split(';', expand=True)
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1, treating 2 as 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x >= 1 else 0)

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Saves the cleaned data frame to a SQLite database.

    Args:
    df (DataFrame): The cleaned data frame to save.
    database_filename (str): The filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to run data processing steps: loading, cleaning, and saving data.
    """
    if len(sys.argv) == 4:
        # Unpack command-line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Load data from the specified file paths
        print(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)

        # Clean the loaded data
        print("Cleaning data...")
        df = clean_data(df)

        # Save the cleaned data to a database
        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")
    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )

if __name__ == "__main__":
    main()
