from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/student_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all form inputs
    features = [
        float(request.form['attendance']),
        float(request.form['study_hours']),
        float(request.form['gpa']),
        float(request.form['test_score'])
    ]

    # Convert to numpy array for model
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)[0]

    if prediction == 1:
        result = "PASS"
    else:
        result = "FAIL"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
