import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#import streamlit as st
import simplejson as json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from tensorflow import keras
from keras_preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
import streamlit as st

MAX_SEQUENCE_LENGTH = 1000
# This is fixed.
EMBEDDING_DIM = 200
roc_t = 0.36430258

with open('C:/Users/falco/Desktop/directory/wildlife_syndrome_app/models/keras_tokenizer_v3_08_21_2022.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('C:/Users/falco/Desktop/directory/wildlife_syndrome_app/models/model_v3_08_21_2022.h5')


st.write("# Wildlife Clinical Classification App")

message_text = st.text_input("Enter text describing clincal evaluation, reason/s for admission")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
labels = ['Clinically healthy', 'Dermatologic disease', 'Gastrointestinal disease', 'Hematologic disease', 'Neurologic disease', 'Nonspecific', 'Nutritional disease', 'Ocular disease', 'Physical injury', 'Respiratory disease', 'Urogenital disease']



def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def predict_conditions(new_case):
    new_case = [new_case]
    #print(new_case)
    seq = tokenizer.texts_to_sequences(new_case)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    sorted_pred = sorted(pred[0])
    if sorted_pred[-1]-sorted_pred[-2] >= roc_t:
        #print ("Final Clinical Classifiaction Prediction")
        #print(labels[np.argmax(pred)])
        clinical_class_1 = labels[np.argmax(pred)] 
        clinical_class_2 = None
    else:
        #print("First Prediction")
        #print(labels[np.argmax(pred)])
        #print("")
        #print("Second Prediction")
        #print(labels[list(pred[0]).index(sorted_pred[-2])])
        clinical_class_1 = labels[np.argmax(pred)]
        clinical_class_2 = labels[list(pred[0]).index(sorted_pred[-2])]
    return clinical_class_1, clinical_class_2

def preprocess_and_predict_input_data(new_case_description):
    nc = clean_text(text = new_case_description)
    cc1, cc2 = predict_conditions(nc)
    return cc1, cc2

if message_text != '':
    result = preprocess_and_predict_input_data(message_text)
    st.write(result)

#from lime.lime_text import LimeTextExplainer
#import streamlit.components.v1 as components

#explain_pred = st.button('Explain Predictions')


#if explain_pred:
#    with st.spinner('Generating explanations'):
#        class_names = labels
#        explainer = LimeTextExplainer(class_names=class_names)
#        exp = explainer.explain_instance(message_text, model.predict, 
#                                         num_features=10)
#        components.html(exp.as_html(), height=800)



