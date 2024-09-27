import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

# Load the trained XGBoost model
model = joblib.load('models/xgb_model.pkl')

# Load the datasets
X_train = pd.read_csv('datasets/X_train.csv')
y_train = pd.read_csv('datasets/y_train.csv')
X_test = pd.read_csv('datasets/X_test.csv')
y_test = pd.read_csv('datasets/y_test.csv')

# Define the expected feature names based on the model
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

        # Check if the data is a list (for multiple house predictions)
        if isinstance(data, list):
            # Create a DataFrame from the list of inputs
            input_data = pd.DataFrame(data)
            
            # Align the input data with the expected features
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            
            # Predict the sale prices for each house
            predictions = model.predict(input_data).tolist()

            # Return the predictions as JSON
            return jsonify({'predictions': predictions})
        
        else:
            # If it's a single prediction, handle it as before
            input_data = pd.DataFrame([data])
            
            # Align the input data with the expected features
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            
            # Predict the sale price using the model
            prediction = model.predict(input_data)[0]
            
            # Convert the prediction to a Python float (so it can be serialized to JSON)
            prediction = float(prediction)
            
            # Return the prediction as JSON
            return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Route to fetch model performance metrics
@app.route('/model_performance', methods=['GET'])
def model_performance():
    try:
        # Get predictions for both training and testing data
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for training data
        train_r2 = r2_score(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)

        # Calculate metrics for testing data
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        # Return metrics and predictions
        return jsonify({
            'train_r2': train_r2,
            'train_mse': train_mse,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'y_train_pred': y_train_pred.tolist(),
            'y_test_pred': y_test_pred.tolist(),
            'y_train': y_train.squeeze().tolist(),
            'y_test': y_test.squeeze().tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Heroku dynamically assigns a port, so we retrieve it from the environment
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
