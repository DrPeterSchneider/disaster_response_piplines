# import libraries
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

import sys
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''load table from database_filepath into pandas dataframe and
       return the arrays for messages(X), labels(Y) and category names'''
    
    # read data into pandas dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    
    # return arrays for messages, labels and the label names
    X = df['message'].values
    Y = df[df.columns[6:]].values
    category_names = df.columns[6:]
    return X, Y, category_names
                         

def tokenize(text):
    '''normalize and standardize the text argument,
       change the case to lower, strip whitespaces,
       remove stopwords'''
    
    # remove punctuation
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    
    # remove digits
    text = re.sub(' \d+', ' ', text)
    
    # split into tokens
    tokens = word_tokenize(text)
    
    # lemmatize, change case to lower and strip whitespaces
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for token in tokens:
        lemmatized.append(lemmatizer.lemmatize(token).lower().strip())
    
    # remove stopwords
    tokenized_text = [word for word in lemmatized
                      if word not in stopwords.words('english')]
    return tokenized_text


def build_model():
    '''create machine learning model from pipeline after grid search'''
    
    # instantiate machine learning pipeline
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()
                                                      ))
                        ])
    # find model with best parameters
    parameters = {'clf__estimator__min_samples_split': [2, 3]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''return precision, recall, f1-score for each category
        using given model'''
    
    # predict labels with model
    y_pred_cv = model.predict(X_test)
    
    # print classification report for every category
    for i in range(y_pred_cv.shape[1]):
        print(category_names[i].upper())
        print(classification_report(y_pred_cv[:, i], Y_test[:, i]))


def save_model(model, model_filepath):
    '''saves a machine learning model to given filepath'''
    
    # save model to pickle file
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)                                            
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
