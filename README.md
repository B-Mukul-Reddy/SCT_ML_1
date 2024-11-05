SCT_ML_1
SKILLCRAFT TECHNOLOGY INTERNSHIP

House Price Prediction using Linear Regression
Project Overview
This project uses linear regression to predict house prices based on square footage, number of bedrooms, full bathrooms, and half bathrooms. We use two datasets: one containing the features of the houses (square footage, number of bedrooms, etc.) and the other containing the target variable (house prices). The project demonstrates data preprocessing, splitting data into training and testing sets, training the linear regression model, and evaluating its performance using metrics such as Mean Squared Error (MSE) and R-squared (R²).

The project steps include:
Data Loading: Load datasets containing features and house prices.
Data Cleaning: Handle missing values and select relevant features.
Data Splitting: Split the data into training and testing sets.
Model Training: Train a linear regression model on the training set.
Evaluation: Evaluate the model using various metrics and visualize the results.
Visualization: Create plots for model performance and feature importance.
Project Structure
|-- project_folder/
|   |-- test.csv               # Features dataset
|   |-- sample_submission.csv  # Target dataset (SalePrice)
|   |-- house_price_prediction.ipynb  # Jupyter notebook file
|   |-- README.md              # Project documentation
Requirements
Before running the project, make sure you have the following Python libraries installed:

pandas for data manipulation
scikit-learn for machine learning models and metrics
You can install the required packages using pip:

pip install pandas scikit-learn
