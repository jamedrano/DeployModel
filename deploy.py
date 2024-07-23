import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from io import BytesIO

# Function to load the pickled model
def load_model(file):
    try:
        model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to predict and add predictions to the dataset
def predict(model, dataset):
    try:
        predictions = model.predict(xgb.DMatrix(dataset))
        dataset['Predictions'] = predictions
        return dataset
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None

# Streamlit interface
st.title("XGBoost Model Predictor")

# Upload model
st.subheader("Upload a trained XGBoost model in pickle format")
model_file = st.file_uploader("Choose a model file", type="pkl")
if model_file:
    model = load_model(model_file)

# Upload dataset
st.subheader("Upload a dataset")
data_file = st.file_uploader("Choose a CSV file", type="csv")
if data_file:
    dataset = pd.read_csv(data_file)

    # Show dataset
    st.write("Dataset Preview:")
    st.write(dataset.head())

    if model:
        # Make predictions
        predicted_dataset = predict(model, dataset)

        if predicted_dataset is not None:
            # Show predictions
            st.write("Predictions:")
            st.write(predicted_dataset.head())

            # Download predicted data
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                predicted_dataset.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)

            st.download_button(
                label="Download Predicted Data",
                data=output,
                file_name="predicted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
