# Disaster Response Pipeline Project
## Introduction
This projet analyze messages sent after disasters to build a model to predict the\n messages' categories. </br>
This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety. </br>
The purpose of this project is to build a supervised machine learning model to better predict the category of 'key words' in messages after disasters. It is benificial for disaster organizaions efficiently extract information quickly relocates resources in the future.

## Contents
This projects includes:
1. ETL pipeline </br>
   `process_data.py` </br>
   `messages.csv`
   `categories.csv`
   `DisasterResponse.db`
2. ML pipeline </br>
    `train_classifier.py` </br>
    `classifier.pkl`
3. App </br>
   A flask based web app to categorize new messages and visualize the training set genres, categories and social media categories. </br>


## Dependencies
- Python 3
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web App: Flask, Plotly

## QuickStart
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
     - `python run.py`
     - Go to http://0.0.0.0:3001/

## Acknowledgement
The data is provided by Figure Eight.
