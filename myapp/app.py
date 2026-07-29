import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from weather_data import predict, get_model_info

app = Flask(__name__, template_folder='templates')

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
        result = predict(body)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info')
def api_model_info():
    return jsonify(get_model_info())

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f'Starting server at http://127.0.0.1:{port}  (press Ctrl+C to stop)')
    app.run(debug=True, host='0.0.0.0', port=port)
