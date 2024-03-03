from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

app = FastAPI()

# Load the saved model
model = joblib.load(r'C:\Users\krisi.afezolli\Desktop\personal_project\visable_coding_challenge\linear_svc_model.joblib')

# Load the data[text] column vectorizer
vectorizer = joblib.load(r'C:\Users\krisi.afezolli\Desktop\personal_project\visable_coding_challenge\tfidf_vectorizer_training.joblib')

# Create a Pydantic model for input validation
class InputText(BaseModel):
    text: str

@app.post("/nlp-predictions")
def predict(input_text: InputText):
    # Vectorize the input text using the loaded vectorizer
    input_vectorized = vectorizer.transform([input_text.text])

    # Make predictions
    prediction = model.predict(input_vectorized)
    return {"prediction": prediction[0]}
