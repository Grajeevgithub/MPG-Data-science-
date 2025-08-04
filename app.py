from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = tf.keras.models.load_model("mpg.h5")

# Define the scaler with the same columns used in training
scaler = StandardScaler()
data = pd.read_csv("mpg.csv")
X = data.drop(columns=['mpg', 'car name'])  # Adjust as per your training
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = scaler.transform([features])
        prediction = model.predict(final_input)[0][0]
        return render_template('index.html', prediction_text=f"Predicted MPG: {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
