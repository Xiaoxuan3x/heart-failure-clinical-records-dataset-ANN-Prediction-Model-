import numpy as np
import pandas as pd
import streamlit as st
import keras
from PIL import Image
import pickle
##load ann model
with open(r'D:\Users\V\Desktop\Bootcamp\RegClass\ann_heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

## load the copy of dataset
df=pd.read_csv('heart_failure_clinical_records_dataset.csv')

## set page configuration
st.set_page_config(page_title='Heart Failure Death Event', layout='wide')

## add page title and content
st.title('Heart Failure Death Event Prediction')
st.write('Please input your value of following features, to see if you are patients at high risk of death from cardiovascular disease')

##add image
image=Image.open(r'D:\Users\V\Desktop\Bootcamp\RegClass\heart_failure.jpg')
st.image(image,use_column_width=True)


#get user's input
age = st.number_input('Age', min_value=0, max_value=150)
time = st.number_input('Follow-up period (days)', min_value=0, max_value=300)
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase (mcg/L)', min_value=0, max_value=10000)
ejection_fraction = st.number_input('Ejection Fraction (%)', min_value=0, max_value=100)
platelets = st.number_input('Platelets (kiloplatelets/mL)', min_value=0, max_value=1000000)
serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, max_value=30.0, step=0.1)
serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=0, max_value=200)

sex=st.number_input("Input 0 if you are female, 1 if you are male",min_value=0, max_value=1)
smoking=st.number_input("Input 0 if you don't smoke, 1 if you smoke",min_value=0, max_value=1)
high_blood_pressure=st.number_input("Input 0 if you don't have high blood pressure, 1 if you have high blood pressure.",min_value=0,max_value=1)
diabetes=st.number_input("Input 0 if you don't have diabetes, 1 if you have diabetes",min_value=0, max_value=1)
anaemia=st.number_input("Input 0 if you don't have anaemi, 1 if you have anaemi",min_value=0, max_value=1)
# Combine the input data into a single list
input_array = [[age, anaemia,diabetes, creatinine_phosphokinase, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium,sex,smoking, time]]


# print the prediction result
if st.button("Predict"):
    result = model.predict(input_array)
    updated_res = result.flatten().astype(int)
    if updated_res > 0.5:
        st.write("Cardiovascular disease poses a greater risk of death to you.Please seek regular advice from your doctor")
    else:
        st.write("Cardiovascular disease poses a small risk of death")
