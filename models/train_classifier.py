import sys
import re
import numpy as np
import pandas as pd
import nltk
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
#from sklearn.svm import LinearSVC


def load_data(database_filepath):
    """
        Loads an sqlite database from the given filepath and returns inputs,
        targets and target names 
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_db', engine)

    # create inputs and targets datasets
    X =  df['message']
    Y = df.drop(['id', 'original', 'message', 'genre'], axis=1)

    # create a list of target names for the classification
    category_names = list(Y.columns)
    
    return X, Y, category_names

def tokenize(text):
    """
        Performs text cleaning and returns list of cleaned tokens
    """
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize 
    tokens = word_tokenize(text)

    # create clean tokens using WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
        Builds a ML pipeline and returns
        the GridSearch model object  
    """

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))])

    # create parameters dictionary
    # using less parameters to cut down time
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 1.0),
        #'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 5]
    }
    
    # initialize GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs=-1) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluates the model performance on the test set and
        prints multi-output classification results 
    """
    # predict on the test dataset
    Y_pred = model.predict(X_test)

    # print evaluation results
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
        saves the trained model to the given filepath
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()