from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load our pre-trained h5 model
model = tf.keras.models.load_model('LungCancer.h5')

# Function to process form data 
# def process_form_data(form_data):
#     # Convert gender to numerical value
#     gender = 1 if form_data['gender'] == 'Male' else 2

#     # Extract numerical values from form data
#     data = [int(form_data[key]) for key in form_data.keys()]
#     # Replace gender value with converted gender
#     data[1] = gender

#     # Perform any additional preprocessing if needed

#     # Return the processed data
#     return np.array(data)
def process_form_data(form_data):
    try:
        # Convert gender to numerical value
        gender = 1 if form_data['gender'] == 'Male' else 2

        # Extract numerical values from form data
        data = [int(form_data[key]) for key in form_data.keys() if key != 'gender']
        # Replace gender value with converted gender
        data.insert(0, gender)  # Insert gender value at the beginning of the list

        # Reshape the data to match the expected input shape
        processed_data = np.array(data).reshape(1, -1)

        # Return the processed data
        return processed_data
    except (KeyError, ValueError) as e:
        # Handle missing keys or invalid values gracefully
        print("Error processing form data:", e)
        return None  # Return None to indicate processing failure


# Function to predict outcome based on model predictions
def predict_outcome(processed_data):
    # Make predictions using the pre-trained model
    predictions = model.predict(processed_data)
    
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

        # Process form data
        processed_data = process_form_data(form_data)
        # Predict outcome
        outcome = predict_outcome(processed_data)
        # Print the predicted outcome
        print("Predicted Outcome:", outcome)
        # Render a template with the prediction results
        return render_template('result.html', outcome=outcome)
    else:    
        # If the request method is not POST, redirect to the home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
