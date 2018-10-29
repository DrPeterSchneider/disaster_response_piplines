# Disaster Response Pipeline Project

### Summary:

This project is a webapp for labeling text messages during disasters.
The company FigureEight provided 30,000 text messages from disastrous events which are labeled
with 36 categories. Based on this labeled data a ETL(Extract Transform Load)- and a
ML(Machine Learing)-Pipeline where built as the backend of a webapp. In this webapp you can
enter a message and an algorithm labels the messages with the matching categories out of 36.

### Files:
	- disaster_messages.csv : CSV-file containing text messages in English and in the original
    	language of the message, also a label for the distribution genre for each message
	- disaster_categories.csv : CSV-file containing category labels for each message
    - process_data.py : ETL-pipeline for cleaning, merging and storing the data from CSV-files
    - DisasterResponse.db : database with cleaned data from the CSV-files
    - train_classifier.py : MLP-pipeline which tokenizes the text messages and builds a model
    	after a crossvalidated grid search. The model is evaluated und saved in pickle-file.
	- finanlized_model.pkl : the model from the MLP-pipeline is saved here
	- run.py : file to build and run the webapp with flask and plotly
    - go.html : webapp design especially for the message query
    - master.html: webapp design for the webapp
    

### Instructions:
1. Run the following commands in the project's root directory to set up the database and the model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/finanlized_model.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
