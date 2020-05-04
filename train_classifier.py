import sys
import re
import pickle
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import nltk
nltk.download('stopwords')


def load_data(database_filepath):
    """load data from database
    Arg: database filepath
    Return: features (X), labels (y) and label names (category_names)
    """
    # connect db adn read data
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * from DisasterMessages', engine)

    category_names = df.columns[-36:].tolist()
    X = df['message'].values
    y = df.iloc[:, 4:].values
    return X, y, category_names

def tokenize(text):
    """Tokenize each message
    Arg: messages
    Return: token lists after normalization and lemmatization
    """
    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # tokenize the word into words
    tokens = word_tokenize(text)

    # remove stopwords
    stop_words  = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize the word
    lemmatizer = WordNetLemmatizer()
    clean_token = []
    for token in tokens:
        clean_token.append(lemmatizer.lemmatize(token, pos='v').lower().strip())
    return clean_token

class WordCount(BaseEstimator, TransformerMixin):
    """ Custom transformer to count the number of words in text
    """
    def word_count(self, text):
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
        # tokenize the word into words
        tokens = word_tokenize(text)
        return len(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        counts = pd.Series(X).apply(self.word_count)
        return pd.DataFrame(counts)


def build_model():
    """
    A multi-output Adaboost classifier machine learning pipeline for
    natural language processing with tdidf, wordcount, and  grid search for optimization.
    Returns:
        Fitted Adaboostclassifer.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('count', WordCount()),
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
                ]))
                ])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 4, 6],
        'features__text_pipeline__tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(cv, X_test, y_test, category_names):
    """Print out classification report for each label
    Args:
        Classification Model
        X_test, y_test, Array-like
        category_names
    return: classification report
    """
    for i in range(36):
        print(labels[i])
        print(classification_report(y_test[:,i], y_pred[:,i]))
    print('\nBest Parameters:', cv.best_params_)


def save_model(model, model_filepath):
    """Save fitted Model as pickle file"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
