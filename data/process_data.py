import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''load data from csv-files'''
    
    # load files into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge both dataframes
    df = messages.merge(categories, on='id', how='outer')
    return df


def clean_data(df):
    '''extract column names and label values for categories,
     remove duplicates and irrelevant values'''
    
    # extract column names, built a dataframe for categories
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    category_colnames = [entry[:-2] for entry in row]
    categories.columns = category_colnames
    
    # convert column entries to integers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # merge categories with df
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates('id', inplace=True)
    
    # remove all rows with related = 2, keep only 0,1
    df = df[df['related'] != 2]
        
    return df


def save_data(df, database_filename):
    '''save the cleaned dataframe to Database'''
    
    # delete table if it was created already
    conn = sqlite3.connect(str(database_filename))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS disaster_response")

    # save dataframe to table in database
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
         database_filepath) = sys.argv[1:]

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
