# Heritage Housing Flask API

This Flask API was developed as part of the Heritage Housing project to predict house prices in Ames, Iowa. The API provides a simple interface for interacting with the machine learning model, allowing users to make predictions on house prices based on various input features.

## Table of Contents

1. [Project Summary](#project-summary)
2. [Features](#features)
3. [API Endpoints](#api-endpoints)
4. [Usage](#usage)


## Project Summary

The Heritage Housing Flask API was created to offload the machine learning model from the main Streamlit app, allowing for a more lightweight deployment. This was necessary due to slug size limits on Heroku when deploying large libraries like `xgboost`. 

The API allows users to send POST requests with house feature data and receive predicted house prices. It serves as a backend service that can be integrated with the front-end Streamlit app or other applications.

## Features

- **POST endpoint for predictions**: Users can send house feature data (e.g., `GrLivArea`, `GarageArea`, etc.) and get predicted house prices.
- **Machine learning integration**: The API uses an XGBoost model trained on the Ames Housing dataset to make predictions.
- **Lightweight**: The API is designed for minimal dependencies to ensure it can be deployed efficiently on platforms like Heroku.

## API Endpoints

**`/predict` (POST)**

  - Description: This endpoint receives JSON input of house features and returns a predicted house price.

  - Input: A JSON object with house attributes (e.g., GrLivArea, GarageArea, YearBuilt, etc.).

  Example request body:
  ```
    {
    "GrLivArea": 1500,
    "TotalSF": 2000,
    "GarageArea": 500,
    "YearBuilt": 2000,
    "OverallQual": 7
    }
  ```

  - Response: A JSON object containing the predicted sale price.

  Example response:
  ```
    {
    "prediction": 235000.50
    }
  ```