from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load our pre-trained h5 model
model = tf.keras.models.load_model('Optimised_model.h5')
# Load the scaler object
scaler = joblib.load('scaler.pkl')

# Function to process and scale the form data
def process_and_scale_data(form_data, scaler):
    try:
        # Convert gender to numerical value
        gender = 1 if form_data['gender'] == 'Male' else 2

        # Extract age values from form data
        age = int(form_data['age']) 
        
        # Include age and gender at the beginning
        data = [age, gender]  
        print("Data after inserting gender:", data)

        # Extract other numerical values from form data
        other_data = [int(form_data[key]) for key in form_data.keys() if key not in ['age', 'gender']]
        data.extend(other_data) 
        print("Processed data shape:", data)

        # Reshape the data to match the expected input shape
        processed_data = np.array(data).reshape(1, -1)
        print("Processed data shape:", processed_data.shape)

        # Scale the processed data using the pre-fitted scaler
        scaled_data = scaler.transform(processed_data)

        # Return the scaled data
        return scaled_data
    except (KeyError, ValueError) as e:
        # Handle missing keys or invalid values gracefully
        print("Error processing and scaling form data:", e)
        return None  # Return None to indicate processing failure 


# Function to predict outcome based on model predictions
def predict_outcome(scaled_data, model):
    try:
        # Make predictions using the pre-trained model
        predictions = model.predict(scaled_data)
        
        # Convert predictions to class labels
        predict_classes = np.argmax(predictions, axis=1)

        # Determine the predicted outcome based on the prediction
        if predict_classes[0] == 0:
            outcome = 'Low'
        elif predict_classes[0] == 1:
            outcome = 'Medium'
        else:
            outcome = 'High'
        # Return outcome
        return outcome
    except Exception as e:
        # Handle prediction errors gracefully
        print("Error predicting outcome:", e)
        return None  # Return None to indicate prediction failure

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Form page route
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        form_data = request.form.to_dict()

        # Remove the 'form_title' field from the form data
        form_data.pop('form_title', None)

        # Process and scale form data
        scaled_data = process_and_scale_data(form_data, scaler)
        if scaled_data is None:
            # Return an error message if processing fails
            return render_template('error.html', message="Error processing form data")

        # Print the processed form data for debugging
        print("Processed Form Data:", scaled_data)

        # Predict outcome
        outcome = predict_outcome(scaled_data, model)
        if outcome is None:
            # Return an error message if prediction fails
            return render_template('error.html', message="Error predicting outcome")
       
        # Print the predicted outcome
        print("Predicted Outcome:", outcome)

        # Render a template with the prediction results
        return render_template('result.html', outcome=outcome)
    else:    
        # If the request method is not POST, redirect to the home page
        return redirect('/')    


if __name__ == '__main__':
    app.run(debug=True, port=8080)
