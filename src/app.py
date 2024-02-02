# importacion de paquetes
import streamlit as st
import os
import numpy as np
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import joblib

import warnings
nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings("ignore")

np.random.seed(123)
 
stop_words = stopwords.words("english")

# Función para limpiar el texto
@st.cache_data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Limpiamos el texto quitando StopWords y Lemmatizando
 
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text) # eliminar caracteres especiales
    text = re.sub(r"\'s", " ", text) # quitar apóstrofes 
    text = re.sub(r"http\S+", " link ", text) # quitar enlaces sustituyendolos por "link"
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  
 
    # Quitamos signos de puntuacion
    text = "".join([c for c in text if c not in punctuation])
 
    # Quitamos Stopwords
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
 
    # Lemmatizamos las palabras
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
 
    # Devolvemos una lista de palabras tratadas
    return text

# Función para hacer la predicción
@st.cache_data
def make_prediction(review):
 
    # Limpiamos los datos llamando a la función de limpieza
    clean_review = text_cleaning(review)
 
    # Cargamos el modelo
    model = joblib.load("model/sentiment_review.pkl")
 
    # Hacemos la predicción
    result = model.predict([clean_review])
 
    # Calculamos la probabilidad
    probas = model.predict_proba([clean_review])
    probability = "{:.2f}".format(float(probas[:, result]))
 
    return result, probability


st.title("Aplicación de análisis de sentimientos en las reseñas")
st.write(
    "App de ML que precice el sentimiento de reseñas de películas en Inglés"
)

form = st.form(key="my_form")
review = form.text_input(label="Introduce tu reseña en inglés para su análisis")
submit = form.form_submit_button(label="Descubre")

if submit:
    # Calcula la predicción
    result, probability = make_prediction(review)
 
    # Muestra los resultados 
    st.header("Resultados")
 
    if int(result) == 1:
        st.write("Esto es una reseña positiva con una probabilidad del ", probability)
    else:
        st.write("Esto es una reseña negativa con una probabilidad del ", probability)

