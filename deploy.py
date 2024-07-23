import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from io import BytesIO

def load_model(uploaded_file):
 modelo = pd.read_pickle(uploaded_file)
 return modelo

def load_data(uploaded_file,sh,h):
 data = pd.read_excel(uploaded_file,header=h,sheet_name=sh,engine='openpyxl')
 data.columns = data.columns.str.strip()
 for col in data.columns:
  if data[col].dtype == 'O':
   data[col] = data[col].str.strip()    
 return data

def pegar(df1, df2):
 return pd.concat([df1, df2.set_index(df1.index)], axis=1)

archivoModelo = st.file_uploader("Cargar Modelo")
   

if archivoModelo is not None:
   modeloprod = load_model(archivoModelo)
   st.write("Model loaded")
   st.write(modeloprod)
    
   datosprod = st.file_uploader("Cargar Datos Prod")
   if datosprod is not None:     
      datospred = load_data(datosprod, 'Sheet1', 0) 
     
      st.write("Predicting...")  
      ypred = modeloprod.get_booster().predict(xgb.DMatrix(datospred))
      ypred2 = pd.DataFrame({'ypred':ypred})
      resultados = pegar(datospred, ypred2)
      st.dataframe(resultados)
      resulta2 = to_excel(resultados)
      st.download_button(label='📥Descargar resultados',data=resulta2 ,file_name= 'resultados.xlsx')
   
