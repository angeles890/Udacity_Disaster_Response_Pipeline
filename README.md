# Disaster Response Pipeline Project

### Goals
The overall goal of this project is to build a data pipeline that leverages a wide variety of best practices to ingest and process data and then leverage the raw data using machine learning to develop a NLP model to classify text. The final train model is then deployed as part of a web-app that enables users to type in new text and have the input classified based on the learned model. Additionally, the index page of the web app contains high level visualizations related to the training data.

### Data
The original data was provided by Figure Eight via two CSVs, and it consists of thousands of real messages that were sent during disaster events. Each message has been labeled with the appropriate disaster related category such as “Aid Related”, “Medical Help”, “Food”, etc.
1. disaster_messages.csv - Raw unprocessed messages created by users during disaster events. Each message as a unique identifier: id
2. disaster_categories.csv - File with each message id and string containing every category possible, and a 1 if the category applies to the message and 0 otherwise. 

A review of the data shows a very unbalanced data set with a number of categories getting very few labels relative to other labels. With highly unbalanced data sets, all things being equal, most ML models will simply ‘predict’ the more popular label. There are a number of ways to deal with this such as creating a synthetic dataset that under or over represents sub-samples to create more balanced datasets. Other methods include hyper-parameter tuning for given ML models, specifically taking advantage of the ‘class_weight’ hyper-parameter available to some ML models. For this particular project, we take advantage of ‘class_weight’.

### Code

This project consist of 3 core Python files
1. process_data.py - Contains the code needed to extract, clean and store data from both the CSV files provided by Figure Eight. The final step then creates a SQLite database and stores the final cleaned data to support model development later in the pipeline. 
2. train_classifier.py - Contains the code needed to train, assess, and save a machine learning model built on the cleaned text data created by process_data.py. This file contains the work needed to support text parsing and feature extraction from text to support classification operations. A random forest classifier is trained on the tokenized data. The final model is saved as a pickle file to be called by 'run.py'. NOTE, due to size limitations on GitHub, the actual pickle file was not committed; however, proper running of the scripts as outlined in the 'Run' section below, will create all necessary artifacts to run the web-app and generate predictions.
3. run.py - This file, largely created by Udacity, contains the code needed to render the web app and handle user requests to the classification API (pickled file created by 'train_classifier.py').

### Requirements 
<ul>
    <li>Python 3.6</li>
    <li>SkLearn</li>
    <li>Pandas</li>
    <li>Numpy</li>
    <li>NLTK</li>
    <li>Flask</li>
    <li>Sqlalchemy</li>
    <li>Sys</li>
    <li>Re</li>
    <li>Pickle</li>
    <li>Matplotlib</li>
    <li>Plotly</li>
    <li>Ast</li>
</ul>



### Run (Original instructions as provided by Udacity) :
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
