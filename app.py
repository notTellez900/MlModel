from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import random

app=Flask(__name__)

modelo = load('knn_modelV2.joblib')
transformer = load('transformerV2.joblib')

def apply_transformations(input_data):
    # Realizar las mismas transformaciones que se hicieron durante el entrenamiento del modelo
    transformed_data = input_data.copy()

    # Reemplazo de valores categóricos por números
    transformed_data.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
                              'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1}, inplace=True)

    # Asegurarse de que 'Diabetic' es un entero
    if 'Diabetic' in transformed_data.columns:
        transformed_data['Diabetic'] = transformed_data['Diabetic'].astype(int)

    # Aquí añadirías más transformaciones, como one-hot encoding, estandarización, etc.

    transformed_data = transformer.transform(transformed_data)

    return transformed_data



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.DataFrame([data])

    df_transformed = apply_transformations(df)

    result = modelo.predict(df_transformed)

    result2 = random.choice([0, 1])

    return jsonify({'prediction': result2})

if __name__ == '__main__':
    app.run(debug=True)