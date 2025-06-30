import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Define the datasets and their file paths
datasets = {
    "Maize": "Maize.csv",
    "Masoor": "Masoor.csv",
    "Moong": "Moong.csv",
    "Niger": "Niger.csv",
    "Paddy": "Paddy.csv",
    "Ragi": "Ragi.csv",
    "Rape": "Rape.csv",
    "Safflower": "Safflower.csv",
    "Sesamum": "Sesamum.csv",
    "Soyabean": "Soyabean.csv"
}

# Directory to save models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Function to train and save the model
def train_and_save_model(dataset_name, file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Define features and target
    X = data[['Month', 'Year', 'Rainfall']]
    y = data['WPI']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), ['Month', 'Year', 'Rainfall'])]
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model pipeline
    model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=3)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Print metrics
    print(f"\nDataset: {dataset_name}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save the model
    model_path = os.path.join(model_dir, f"{dataset_name}_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved at: {model_path}")

# Train and save models for all datasets
for name, path in datasets.items():
    train_and_save_model(name, path)

print("\nAll models trained and saved successfully!")
