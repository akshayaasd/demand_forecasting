from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create Flask app
app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('random_forest_model.pkl')

# Route to serve the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and redirect to the result page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        store_id = int(request.form['store_id'])
        sku_id = int(request.form['sku_id'])
        total_price = float(request.form['total_price'])
        base_price = float(request.form['base_price'])
        is_featured_sku = int(request.form['is_featured_sku'])
        is_display_sku = int(request.form['is_display_sku'])

        # Prepare input data for prediction
        input_data = np.array([[store_id, sku_id, total_price, base_price, is_featured_sku, is_display_sku]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Redirect to the result page with the prediction
        return redirect(url_for('result', prediction=round(prediction, 2)))

# Route to display the prediction result
@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
