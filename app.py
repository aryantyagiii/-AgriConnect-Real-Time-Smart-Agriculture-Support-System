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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import razorpay
import uuid


# Initialize Flask app
app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize Razorpay client with environment variables
RAZORPAY_KEY_ID = os.getenv('RAZORPAY_KEY_ID')
RAZORPAY_KEY_SECRET = os.getenv('RAZORPAY_KEY_SECRET')

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("Warning: Razorpay credentials not found in environment variables")
    client = None
else:
    client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

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

def validate_csv_columns(data, required_columns=['Month', 'Year', 'Rainfall', 'WPI']):
    """Validate that the CSV has all required columns"""
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")
    return True

# Function to train and save the model
def train_and_save_model(dataset_name, file_path):
    try:
        # Load dataset
        print(f"\nProcessing dataset: {dataset_name}")
        data = pd.read_csv(file_path)
        
        # Validate columns
        validate_csv_columns(data)
        
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
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Save the model
        model_path = os.path.join(model_dir, f"{dataset_name}_model.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Model saved at: {model_path}")
        
        return True
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        return False

# Train and save models for all datasets
print("Training models...")
successful_models = []
for name, path in datasets.items():
    if train_and_save_model(name, path):
        successful_models.append(name)

print(f"\nSuccessfully trained models: {len(successful_models)}/{len(datasets)}")

# Load trained models
models = {}
for name in successful_models:
    model_path = os.path.join(model_dir, f"{name}_model.pkl")
    if os.path.exists(model_path):
        models[name] = joblib.load(model_path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        data = request.get_json()
        print("Request data:", data)
        
        if not data:
            print("No data provided")
            return jsonify({"error": "No data provided"}), 400
            
        crop = data.get("crop")
        month = data.get("month")
        year = data.get("year")
        rainfall = data.get("rainfall")
        
        print(f"Processing request for crop: {crop}, month: {month}, year: {year}, rainfall: {rainfall}")
        
        if not all([crop, month, year, rainfall]):
            missing = [k for k, v in {'crop': crop, 'month': month, 'year': year, 'rainfall': rainfall}.items() if not v]
            print(f"Missing fields: {missing}")
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
        
        if crop not in models:
            print(f"Model not found for crop: {crop}")
            return jsonify({"error": f"Model not found for crop: {crop}. Available crops: {list(models.keys())}"}), 400
        
        try:
            month = int(month)
            year = int(year)
            rainfall = float(rainfall)
        except ValueError as e:
            print(f"Invalid numeric values: {str(e)}")
            return jsonify({"error": "Invalid numeric values provided"}), 400
            
        if not (1 <= month <= 12):
            print(f"Invalid month: {month}")
            return jsonify({"error": "Month must be between 1 and 12"}), 400
            
        model = models[crop]
        # Create a pandas DataFrame with the input data
        input_data = pd.DataFrame({
            'Month': [month],
            'Year': [year],
            'Rainfall': [rainfall]
        })
        print(f"Making prediction with input data: {input_data}")
        
        prediction = model.predict(input_data)[0]
        print(f"Prediction result: {prediction}")
        
        response = {
            "crop": crop,
            "predicted_WPI": float(prediction),
            "input_data": {
                "month": month,
                "year": year,
                "rainfall": rainfall
            }
        }
        print("Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/api/create-order', methods=['POST'])
def create_order():
    try:
        if not client:
            return jsonify({"error": "Payment service not configured"}), 503
            
        data = request.get_json()
        if not data or 'finalTotal' not in data:
            return jsonify({"error": "Invalid request data"}), 400
            
        amount = int(data['finalTotal'] * 100)  # Convert to paise
        
        # Create Razorpay order
        order_data = {
            'amount': amount,
            'currency': 'INR',
            'receipt': f'order_{uuid.uuid4().hex[:8]}',
            'payment_capture': 1
        }
        
        order = client.order.create(data=order_data)
        
        return jsonify({
            'id': order['id'],
            'amount': order['amount'],
            'currency': order['currency']
        })
        
    except razorpay.errors.BadRequestError as e:
        print(f"Razorpay Bad Request Error: {str(e)}")
        return jsonify({"error": "Invalid payment request"}), 400
    except razorpay.errors.AuthenticationError as e:
        print(f"Razorpay Authentication Error: {str(e)}")
        return jsonify({"error": "Payment service authentication failed"}), 503
    except Exception as e:
        print(f"Error creating order: {str(e)}")
        return jsonify({"error": "Failed to create order"}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
