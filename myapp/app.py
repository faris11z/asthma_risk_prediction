import os
from flask import Flask, request, jsonify, render_template
from weather_data import get_weather
from model_training import predict_pefr, get_model_info

app = Flask(__name__, template_folder='templates')

def _build_result(data, temp, hum, pm2, pm10, predicted_pefr):
    city = data['city'].strip().lower()
    age = int(data['age'])
    height = int(data['height'])
    gender = int(data['gender'])
    smoking = int(data['smoking'])
    asthma = int(data['asthma'])
    actual_pefr = float(data['actual_pefr'])

    ratio = (actual_pefr / predicted_pefr) * 100

    if ratio >= 80:
        zone = 'SAFE'
    elif ratio >= 50:
        zone = 'MODERATE'
    else:
        zone = 'RISK'

    return {
        'city': city,
        'user': {
            'age': age,
            'height': height,
            'gender': 'Male' if gender == 1 else 'Female',
            'smoking': 'Yes' if smoking else 'No',
            'asthma_history': 'Yes' if asthma else 'No',
            'actual_pefr': actual_pefr,
        },
        'environment': {
            'temperature': temp,
            'humidity': hum,
            'pm25': pm2,
            'pm10': pm10,
        },
        'result': {
            'predicted_pefr': round(predicted_pefr),
            'ratio': round(ratio, 1),
            'zone': zone,
        }
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    body = request.get_json()
    if not body:
        return jsonify({'error': 'Request body is required'}), 400

    required = ['city', 'age', 'height', 'gender', 'smoking', 'asthma', 'actual_pefr']
    for field in required:
        if field not in body or (isinstance(body[field], str) and not body[field].strip()):
            return jsonify({'error': f'{field} is required'}), 400

    try:
        age = int(body['age'])
        if age < 1 or age > 120:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Age must be a number between 1 and 120'}), 400

    try:
        height = int(body['height'])
        if height < 100 or height > 250:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'Height must be a number between 100 and 250 cm'}), 400

    if body['gender'] not in ('0', '1'):
        return jsonify({'error': 'Gender must be 0 (Female) or 1 (Male)'}), 400
    if body['smoking'] not in ('0', '1'):
        return jsonify({'error': 'Smoking must be 0 (No) or 1 (Yes)'}), 400
    if body['asthma'] not in ('0', '1'):
        return jsonify({'error': 'Asthma history must be 0 (No) or 1 (Yes)'}), 400

    try:
        pefr = float(body['actual_pefr'])
        if pefr <= 0 or pefr > 1000:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'PEFR must be a positive number between 1 and 1000'}), 400

    try:
        city = body['city'].strip().lower()
        temp, hum, pm2, pm10 = get_weather(city)
        features = [int(body['age']), int(body['height']), int(body['gender']),
                    int(body['smoking']), int(body['asthma']), temp, hum, pm2, pm10]
        predicted_pefr = predict_pefr(features)
        result = _build_result(body, temp, hum, pm2, pm10, predicted_pefr)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info')
def api_model_info():
    return jsonify(get_model_info())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Starting server at http://127.0.0.1:{port}  (press Ctrl+C to stop)')
    app.run(debug=True, host='0.0.0.0', port=port)
