# Disaster Response Pipeline Project

In this Project, we use a data set from Figure Eight containing real messages that were sent during disaster events. We create a machine learning pipeline to categorize these events so the messages could be send to an appropriate disaster relief agency.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Structure of the project
- app
  + template
  + master.html  # main page of web app
  + go.html  # classification result page of web app
  +run.py  # Flask file that runs app

- data
  + disaster_categories.csv  # data to process 
  + disaster_messages.csv  # data to process
  + process_data.py 
  + DisasterResponse.db   # database to save clean data to

- models
  + train_classifier.py
  + classifier.pkl  # saved model 

- notebooks
  + ETL Pipeline Preparation.ipynb # ETL pipeline exploration
  + ML Pipeline Preparation .ipynb # ML pipeline exploration

- README.md
