from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained XGBoost model
model = joblib.load('models/xgb_model.pkl')

# Define the expected feature names based on your model
expected_features = ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType1', 
                     'BsmtUnfSF', 'EnclosedPorch', 'GarageArea', 'GarageFinish', 'GarageYrBlt', 'GrLivArea', 
                     'KitchenQual', 'LotArea', 'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 
                     'OverallQual', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'TotalSF']

@app.route('/')
def home():
    return "House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Create a DataFrame from the input data
        input_data = pd.DataFrame([data])

        # Align the input data with the expected features
        # Fill missing features with default values (e.g., 0 for numeric fields)
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        # Predict the sale price using the model
        prediction = model.predict(input_data)[0]

        # Convert the prediction to a Python float (so it can be serialized to JSON)
        prediction = float(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
