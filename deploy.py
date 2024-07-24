import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from io import BytesIO

# Function to load the pickled model
def load_model(uploaded_file):
    try:
        model = pd.read_pickle(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to load the dataset
def load_data(uploaded_file, sh, h):
    try:
        data = pd.read_excel(uploaded_file, header=h, sheet_name=sh, engine='openpyxl')
        data.columns = data.columns.str.strip()
        for col in data.columns:
            if data[col].dtype == 'O':
                data[col] = data[col].str.strip()    
        return data
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

# Function to concatenate dataframes
def pegar(df1, df2):
    return pd.concat([df1, df2.set_index(df1.index)], axis=1)

# Function to convert dataframe to excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Streamlit interface
st.title("Predictor de Resistencia a la Tensión")

# Upload models
st.sidebar.header("Cargar los modelos previamente entrenados")
st.sidebar.subheader("Cargar Modelos en el orden 1D, 3D, 7D y 28D")
model_files = [st.sidebar.file_uploader(f"Cargar el modelo {i}", type="pkl") for i in ["1D", "3D", "7D", "28D"]]

# Check if all model files are uploaded
if all(model_files):
    models = [load_model(file) for file in model_files]

    # Check if all models are loaded
    if all(models):
        st.write("Modelos cargados correctamente")

        # Upload dataset
        st.subheader("Cargar los datos para la predicción")
        datosprod = st.file_uploader("Cargar Datos Prod", type="xlsx")
        if datosprod is not None:
            datospred1 = load_data(datosprod, 'Sheet1', 0)
            if datospred1 is not None:
                try:
                    # 1D
                    ypred = models[0].get_booster().predict(xgb.DMatrix(datospred1))
                    ypred2 = pd.DataFrame({'R1D': ypred})
                    resultados1D = pegar(datospred1, ypred2)

                    datospred3 = resultados1D.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]

                    # 3D
                    ypred = models[1].get_booster().predict(xgb.DMatrix(datospred3))
                    ypred2 = pd.DataFrame({'R3D': ypred})
                    resultados3D = pegar(datospred3, ypred2)

                    datospred7 = resultados3D.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9]]

                    # 7D
                    ypred = models[2].get_booster().predict(xgb.DMatrix(datospred7))
                    ypred2 = pd.DataFrame({'R7D': ypred})
                    resultados7D = pegar(datospred7, ypred2)

                    datospred28 = resultados7D.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10]]

                    # 28D
                    ypred = models[3].get_booster().predict(xgb.DMatrix(datospred28))
                    ypred2 = pd.DataFrame({'R28D': ypred})
                    resultados28D = pegar(datospred28, ypred2)

                    st.dataframe(resultados28D)
                    resulta2 = to_excel(resultados28D)
                    st.download_button(label='📥 Descargar resultados', data=resulta2, file_name='resultados.xlsx')
                except Exception as e:
                    st.error(f"Error processing the data: {e}")
    else:
        st.error("Error loading one or more models")
else:
    st.info("Please upload all four models.")
