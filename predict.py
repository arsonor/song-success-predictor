from joblib import load
import pandas as pd

from flask import Flask, request, jsonify


model_file = 'hit-model.joblib'

with open(model_file, 'rb') as f_in:
    saved_objects = load(f_in)

preprocessor = saved_objects['preprocessor']
model = saved_objects['model']

app = Flask('hit')


@app.route('/predict', methods=['POST'])
def predict():
    song = request.get_json()
    
    X = pd.DataFrame([song])

    X_transformed = preprocessor.transform(X)
    y_pred = model.predict_proba(X_transformed)[:, 1]
    hit = y_pred > 0.37
    
    result = {
        'hit_probability': float(y_pred[0]),
        'hit': bool(hit[0]),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    