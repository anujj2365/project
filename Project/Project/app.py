from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('D:/Project/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        age = int(request.form['age'])  # Ensure age is an integer
        height = int(request.form['height'])  # Ensure height is an integer
        weight = int(request.form['weight'])  # Ensure weight is an integer
        duration = int(request.form['duration'])  # Ensure duration is an integer
        heart_rate = int(request.form['heart_rate'])  # Ensure heart_rate is an integer
        body_temp = int(request.form['body_temp'])  # Ensure body_temp is an integer

        # Encode gender (Male: 0, Female: 1)
        gender = 0 if gender == 'Male' else 1

        # Create input array
        input_features = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

        # Predict using the model
        prediction = model.predict(input_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Calories Burnt: {output} kcal')

if __name__ == '__main__':
    app.run(debug=True)
