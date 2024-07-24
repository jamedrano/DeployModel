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

def to_excel(df):
 output = BytesIO()
 writer = pd.ExcelWriter(output, engine='xlsxwriter')
 df.to_excel(writer, index=False, sheet_name='Sheet1')
 workbook = writer.book
 worksheet = writer.sheets['Sheet1']
 format1 = workbook.add_format({'num_format': '0.00'}) 
 worksheet.set_column('A:A', None, format1)  
 writer.close()
 processed_data = output.getvalue()
 return processed_data

# archivoModelo = st.file_uploader("Cargar Modelos en el orden 1D, 3D, 7D y 28D")
modelos = ["1D", "3D", "7D", "28D"]
st.write("Cargar Modelos en el orden 1D, 3D, 7D y 28D")
model_files = [st.file_uploader(f"Cargar el modelo {i}", type="pkl") for i in modelos]   

if model_files:
   modeloprod1D = load_model(model_files[0])
   modeloprod3D = load_model(model_files[1])
   modeloprod7D = load_model(model_files[2])
   modeloprod28D = load_model(model_files[3])
 
   st.write("Model loaded")
   # st.write(modeloprod)
    
   datosprod = st.file_uploader("Cargar Datos Prod")
   if datosprod is not None:     
      datospred1 = load_data(datosprod, 'Sheet1', 0) 
     
      ## 1D 
      ypred = modeloprod1D.get_booster().predict(xgb.DMatrix(datospred1))
      ypred2 = pd.DataFrame({'R1D':ypred})
      resultados1D = pegar(datospred1, ypred2)
          
      datospred3 = resultados1D.iloc[:, [0,1,2,3,4,5,6,7,9,8]]          

      ## 3D 
      ypred = modeloprod3D.get_booster().predict(xgb.DMatrix(datospred3))
      ypred2 = pd.DataFrame({'R3D':ypred})
      resultados3D = pegar(datospred3, ypred2)
          
      datospred7 = resultados3D.iloc[:, [0,1,2,3,4,5,6,7,8,10,9]]          

      ## 7D 
      ypred = modeloprod7D.get_booster().predict(xgb.DMatrix(datospred7))
      ypred2 = pd.DataFrame({'R7D':ypred})
      resultados7D = pegar(datospred7, ypred2)
          
      datospred28 = resultados7D.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,10]]          

      ## 28D 
      ypred = modeloprod28D.get_booster().predict(xgb.DMatrix(datospred28))
      ypred2 = pd.DataFrame({'R28D':ypred})
      resultados28D = pegar(datospred28, ypred2)
          
     
    
      st.dataframe(resultados28D)
      resulta2 = to_excel(resultados28D)
      st.download_button(label='📥Descargar resultados',data=resulta2 ,file_name= 'resultados.xlsx')
   
