## End-to-end machine learning process for text classification

### Project documentation

This document outlines the development of an end-to-end machine learning process for text classification, focusing on a dataset with German text and six unique labels. 
The process includes data loading, exploration, preprocessing, using TF-IDF vectorization, model selection, evaluation and use case prediction. 

#### Step 1 - Create an End-to-End machine learning process for text classification

The dataset is loaded from a CSV file, named sample_data_for_task1.csv. 
It includes a text column and a label column, where the text represents the content and the label represents the category.

#### Text exploration
Initial exploration involved checking the dataset's structure, identifying unique labels, and visualizing label occurrences. 
Data cleaning steps included handling missing values and removing numeric rows.
These rows contained only 1% of the dataset so dropping them will not be an issue. 
Further analysis involved checking for and handling punctuations as they will later be handled by TF-IDF.

#### Text preprocessing
Text preprocessing involves vectorizing the text using the TF-IDF (Term Frequency-Inverse Document Frequency) method.  
The most correlated unigrams and bigrams are displayed along with example texts. 
TF-IDF is implemented and the column text is tokenized in German, stopwords are removed, and a WordCloud is generated to visualize the most frequent words accross the dataset.

#### Multiclassification models
Continuing with the train_test_split from Scikit-learn we split the dataset into training and testing sets (with a 75% - 25% split). 
Several classification models (LogisticRegression, RandomForest, LinearSVC, and Multinomial NaiveBayes) are evaluated using cross-validation and the mean accuracy to compare model performance.
After metrics evaluations, we implemented k-fold cross-validation (with k=5) on each model using accuracy as the evaluation metric. 
The results,are stored in a pd DataFrame for the sake of structuring/visualization.

#### Model training and evaluation
The Linear Support Vector Machine (LinearSVC) model is chosen for training based on its mean accuracy of 85% compared to other models.
Later, the model is trained on the training set, and its performance is evaluated using a classification report. 
- From the classification report, we can observe that the classes that have a greater number of occurrences tend to have a good f1-score compared to other classes. 
- The categories which yield better classification results are `cnc`, `ft` and `ct`. 

#### Prediction on unseen data
Finally, the model is used to predict labels for unseen data. 
Examples of text predictions are provided for various input cases.

#### Further recommendations
Recommendation on `Text Pre-processing step`: 
1. We might one to explore additional text preprocessing steps, such as stemming or lemmatization. They can help in reducing words to their root form and might improve the generalization of the chosen model.
2. Collecting more data: If possible, we might want to increase the size of our dataset as that can often lead to better model performance.
3. Also, experimenting with other algorithms might be beneficial after having identified their strengths and weaknesses.

### Step 2 - Implement Fast API that serves predictions with your model from step 1.

This documentation explains the FastAPI application developed for making NLP predictions based on a pre-trained machine learning model we created and saved on a specific path. 
The application utilizes the FastAPI framework to create a simple web service that classifies input text into predefined categories like >> `feature: zucker ==> label: ft`.

**File structure:**
main.py: This script contains the FastAPI application code, model loading, and prediction.

linear_svc_model.joblib: This file stores the pre-trained Linear Support Vector Machine (LinearSVC) model using the joblib library.

Regarding dependencies and libraries we have installed:

- fastapi: The FastAPI framework for building APIs.
- uvicorn: A lightweight ASGI server for running FastAPI applications.
- joblib: Used for loading the pre-trained machine learning model.
- scikit-learn: Provides tools for machine learning tasks.
- nltk: Natural Language Toolkit for language processing tasks.

- In order to be able to utilize the model, you would need to send a POST request to the following endpoint >> /nlp-predictions.
- The post request that you'll need to sent will have to look like the following:

curl -X 'POST' \
  'http://127.0.0.1:8000/nlp-predictions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "zucker"}'

The response that you get out of the request, will contain the label classification that the model will make, based on the text property that you send via POST, example: {"classification_result": "ft"}
