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
        # Ensure the dataset is in the correct format
        dmatrix = xgb.DMatrix(dataset)
        predictions = model.predict(dmatrix)
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
st.subheader("Upload a dataset (Excel file)")
data_file = st.file_uploader("Choose an Excel file", type="xlsx")
if data_file:
    dataset = pd.read_excel(data_file)
    
    # Show dataset
    st.write("Dataset Preview:")
    st.write(dataset.head())

    if model:
        # Ensure dataset contains only numeric data for prediction
        numeric_dataset = dataset.select_dtypes(include=[pd.np.number])

        # Make predictions
        predicted_dataset = predict(model, numeric_dataset)

        if predicted_dataset is not None:
            # Combine predictions with original data
            combined_dataset = pd.concat([dataset, predicted_dataset['Predictions']], axis=1)

            # Show predictions
            st.write("Predictions:")
            st.write(combined_dataset.head())

            # Download predicted data
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                combined_dataset.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)

            st.download_button(
                label="Download Predicted Data",
                data=output,
                file_name="predicted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
