import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data
train_data = pd.read_csv('C:/Users/namik/OneDrive/Desktop/SCT/train.csv')

# Feature Engineering: Adding TotalBath and TotalRooms
train_data['TotalBath'] = train_data['FullBath'] + train_data['HalfBath']
train_data['TotalRooms'] = train_data['BedroomAbvGr'] + train_data['TotalBath']

# Log transform the SalePrice to reduce skewness
train_data['LogSalePrice'] = np.log(train_data['SalePrice'])

# Prepare features and target
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalRooms']]
y_train = train_data['LogSalePrice']

# Split the data for internal validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize models with parameters for better generalization
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),  # L2 regularization
    'Lasso': Lasso(alpha=0.1)   # L1 regularization
}

# Train and evaluate each model, including RMSE calculation for interpretability
for name, model in models.items():
    model.fit(X_train_split, y_train_split)
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(np.exp(y_val), np.exp(y_val_pred))  # Undo log transform for MSE
    rmse = np.sqrt(mse)
    print(f"{name} - Validation RMSE: {rmse}")

# Cross-validation with Ridge model to ensure stability across folds
best_model = Ridge(alpha=1.0)  # Ridge usually performs better with multi-features
cross_val_mse = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
cross_val_rmse = np.sqrt(-cross_val_mse).mean()
print(f"Ridge Cross-validated RMSE: {cross_val_rmse}")

# Train best model on the full training set
best_model.fit(X_train, y_train)

# Load test data and apply same transformations
test_data = pd.read_csv('C:/Users/namik/OneDrive/Desktop/SCT/test.csv')
test_data['TotalBath'] = test_data['FullBath'] + test_data['HalfBath']
test_data['TotalRooms'] = test_data['BedroomAbvGr'] + test_data['TotalBath']
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalRooms']]

# Predict on test data and undo log transformation
test_data['PredictedPrice'] = np.exp(best_model.predict(X_test))

# Save the predictions
output_file = 'C:/Users/namik/OneDrive/Desktop/SCT/predicted.csv'
test_data[['Id', 'PredictedPrice']].to_csv(output_file, index=False)

print(f"Predictions saved to '{output_file}'")
