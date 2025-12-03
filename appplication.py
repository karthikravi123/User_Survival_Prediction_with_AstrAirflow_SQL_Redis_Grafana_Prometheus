import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from alibi_detect.cd import KSDrift
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler
##scaling down all refernce data and current data compare them easily it is optional
from src.logger import get_logger
from prometheus_client import start_http_server,Counter,Gauge



logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")

##to detect how many time drift is detected ,how many times predition is made
prediction_count = Counter('prediction_count',"Number of preditcion count")
drift_count = Counter('Drift count','no of times data drift is detected')



MODEL_PATH = "artifacts/models/random_forest_model.pkl"

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin',
                 'Title', 'Pclass_Fare', 'Age_Fare', 'Pclass']


feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_ref_data():
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)
    all_features_df = pd.DataFrame.from_dict(all_features,orient='index')[FEATURE_NAMES]
    scaler.fit(all_features_df)

    return scaler.transform(all_features_df)

historical_data = fit_scaler_on_ref_data()

ksd = KSDrift(x_ref=historical_data,p_val=0.05)


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

        ###data drift detection
        feature_scaled = scaler.transform(features)

        drift = ksd.predict(feature_scaled)
        print("Drift response",drift)
        drift_response = drift.get('data',{})
        is_drift = drift_response.get('is_drift',None)

        if is_drift is not None and is_drift ==1 :
            print("Drift Detected")
            logger.info("Drift Detected")
            #default =1 
            drift_count.inc()


        # Predict
        prediction = model.predict(features)[0]
        prediction_count.inc()
        result = 'Survived' if prediction == 1 else 'Did Not Survive'


        return render_template('index.html', prediction_text=f"The prediction is: {result}")

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response
    return Response(generate_latest(),content_type='text/plain')

if __name__ == "__main__":
    ##default route for metrics connection
    start_http_server(8000)
    app.run(debug=True,host='0.0.0.0',port=5000)
