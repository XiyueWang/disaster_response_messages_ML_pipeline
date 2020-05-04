import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ load message and categories file, merge and store in a dataframe
    Args:
    filepath of messages.csv  and categories.csv
    Return:
    merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return categories, df

def clean_data(categories, df):
    """ clean the merged dataframe
    Arg:
    data(df)
    Return:
    cleaned df with seperate columns for each category and no duplicates
    """
    # split category into seperate columns
    categories = categories.categories.str.split(';', expand=True)
    # extract category names
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories.columns = category_colnames

    # Keep category value only
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # generate cleaned df
    df = pd.concat([df, categories], axis=1)
    df.drop('categories', axis=1, inplace=True)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # remove null columns
    df = df.dropna(subset=['related'])
    return df

def save_data(df, database_filepath):
    """ save cleaned dataframe into sqlAchemy database
    Args:
    df, database name
    Return:
    saved db file
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('DisasterMessages', engine, if_exists='replace', index=False)


def main():
    print(sys.argv)
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
        categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(categories, df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\

              'datasets as the first and second argument respectively, as '\

              'well as the filepath of the database to save the cleaned data '\

              'to as the third argument. \n\nExample: python process_data.py '\

              'disaster_messages.csv disaster_categories.csv '\

              'DisasterResponse.db')

if __name__ == '__main__':

    main()
