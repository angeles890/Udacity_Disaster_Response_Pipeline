# Disaster Response Pipeline Project

### Goals
The overall goal of this project is to build a data pipeline that leverages a wide variety of best practices to ingest and process data and then leverage the raw data using machine learning to develop a NLP model to classify text. The final train model is then deployed as part of a web-app that enables users to type in new text and have the input classified based on the learned model. Additionally, the index page of the web app contains high level visualizations related to the training data.

### Data
The original data was provided by Figure Eight via two CSVs, and it consists of thousands of real messages that were sent during disaster events. Each message has been labeled with the appropriate disaster related category such as “Aid Related”, “Medical Help”, “Food”, etc.
1. disaster_messages.csv - Raw unprocessed messages created by users during disaster events. Each message as a unique identifier: id
2. disaster_categories.csv - File with each message id and string containing every category possible, and a 1 if the category applies to the message and 0 otherwise. 




### Run (Original instructions as provided by Udacity) :
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
