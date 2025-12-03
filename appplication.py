import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")
MODEL_PATH = "artifacts/models/random_forest_model.pkl"

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin',
                 'Title', 'Pclass_Fare', 'Age_Fare', 'Pclass']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # Get input values
        Age = float(data["Age"])
        Fare = float(data["Fare"])
        Pclass = int(data["Pclass"])
        Sex = int(data["Sex"])
        Embarked = int(data["Embarked"])
        Familysize = int(data["Familysize"])
        Isalone = int(data["Isalone"])
        HasCabin = int(data["HasCabin"])
        Title = int(data["Title"])

        # Compute derived features instead of asking user to enter
        Pclass_Fare = Pclass * Fare
        Age_Fare = Age * Fare

        # Create DataFrame
        features = pd.DataFrame([[Age, Fare, Sex, Embarked, Familysize, Isalone, HasCabin,
                                  Title, Pclass_Fare, Age_Fare, Pclass]], columns=FEATURE_NAMES)

        # Predict
        prediction = model.predict(features)[0]
        result = 'Survived' if prediction == 1 else 'Did Not Survive'

        return render_template('index.html', prediction_text=f"The prediction is: {result}")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
