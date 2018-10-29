import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath='disaster_messages.csv',
              categories_filepath='disaster_categories.csv'):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='outer')
    return df


def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = [entry[:-2] for entry in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates('id', inplace=True)
    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename='DisasterResponse.db'):
    conn = sqlite3.connect(database_filename)

    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS disaster_response")
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
