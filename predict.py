import pickle

from flask import Flask, request, jsonify


model_file = 'hit-model.bin'

with open(model_file, 'rb') as f_in:
    saved_objects = pickle.load(f_in)

preprocessor = saved_objects['preprocessor']
model = saved_objects['model']

app = Flask('hit')


@app.route('/predict', methods=['POST'])
def predict():
    song = request.get_json()

    X = preprocessor.transform([song])
    y_pred = model.predict_proba(X)[:, 1]
    hit = y_pred > 0.37
    
    result = {
        'hit_probability': float(y_pred),
        'hit': bool(hit),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    