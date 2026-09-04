import gradio as gr
from weather_data import get_weather
from model_training import predict_pefr, get_model_info

CITIES = [
    'chennai', 'coimbatore', 'madurai', 'tiruchirappalli', 'salem',
    'tirunelveli', 'tiruppur', 'erode', 'vellore', 'thoothukudi',
    'dindigul', 'thanjavur', 'ranipet', 'sivakasi', 'karur',
    'ooty', 'hosur', 'nagercoil', 'kumbakonam', 'cuddalore',
    'kanyakumari', 'ambur', 'nagapattinam', 'bengaluru', 'bangalore',
    'hyderabad', 'mumbai', 'pune', 'delhi', 'kolkata',
    'ahmedabad', 'jaipur', 'lucknow', 'surat',
    'london', 'new york', 'tokyo', 'paris', 'berlin',
    'sydney', 'dubai', 'singapore', 'bangkok', 'kuala lumpur',
    'dhaka', 'colombo', 'kathmandu', 'moscow', 'beijing',
    'seoul', 'cairo', 'istanbul', 'rio de janeiro', 'cape town',
    'los angeles', 'chicago', 'toronto', 'melbourne', 'hong kong',
    'ho chi minh', 'jakarta', 'manila', 'karachi', 'lagos', 'nairobi',
]


def predict(city, age, height, gender, smoking, asthma, actual_pefr):
    city = city.strip().lower()
    if not city:
        return 'Enter a city', '', '', '', '', '', '', ''

    temp, hum, pm2, pm10 = get_weather(city)
    features = [age, height, gender, smoking, asthma, temp, hum, pm2, pm10]
    predicted_pefr = predict_pefr(features)
    ratio = (actual_pefr / predicted_pefr) * 100

    if ratio >= 80:
        zone = 'SAFE'
    elif ratio >= 50:
        zone = 'MODERATE'
    else:
        zone = 'RISK'

    info = get_model_info()
    gender_label = 'Male' if gender == 1 else 'Female'
    smoking_label = 'Yes' if smoking else 'No'
    asthma_label = 'Yes' if asthma else 'No'

    summary = (
        f'**Zone:** {zone}\n\n'
        f'| Metric | Value |\n|---|---|\n'
        f'| Predicted PEFR | {round(predicted_pefr)} L/min |\n'
        f'| Your Actual PEFR | {actual_pefr} L/min |\n'
        f'| PEFR Ratio | {round(ratio, 1)}% |\n'
        f'| Age | {age} years |\n'
        f'| Height | {height} cm |\n'
        f'| Gender | {gender_label} |\n'
        f'| Smoking | {smoking_label} |\n'
        f'| Asthma History | {asthma_label} |\n'
    )

    env = (
        f'| Condition | Value |\n|---|---|\n'
        f'| Temperature | {temp} °C |\n'
        f'| Humidity | {hum}% |\n'
        f'| PM 2.5 | {pm2} µg/m³ |\n'
        f'| PM 10 | {pm10} µg/m³ |\n'
    )

    model_info = (
        f'Model: {info["model"]} | '
        f'Trained on {info["rows"]:,} records | '
        f'{len(info["features"])} features'
    )

    return zone, summary, env, model_info


with gr.Blocks(title='Asthma Risk Prediction') as demo:
    gr.Markdown('''
    # Asthma Risk Prediction 🫁
    Assess your respiratory risk by comparing your actual Peak Expiratory Flow Rate (PEFR) against a machine-learning-predicted healthy baseline.
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### Personal Information')
            city = gr.Textbox(label='City', placeholder='Type a city name...', value='chennai')
            age = gr.Slider(12, 80, value=25, step=1, label='Age (years)')
            height = gr.Slider(100, 250, value=170, step=1, label='Height (cm)')
            gender = gr.Radio(choices=[('Male', 1), ('Female', 0)], label='Gender', value=1)
            smoking = gr.Radio(choices=[('Non-Smoker', 0), ('Smoker', 1)], label='Smoking', value=0)
            asthma = gr.Radio(choices=[('No', 0), ('Yes', 1)], label='Asthma History', value=0)
            actual_pefr = gr.Slider(100, 700, value=450, step=5, label='Actual PEFR (L/min)')
            predict_btn = gr.Button('Analyze Risk', variant='primary', size='lg')

        with gr.Column(scale=1):
            gr.Markdown('### Results')
            zone_output = gr.Markdown('')
            summary_output = gr.Markdown('')
            gr.Markdown('### Environmental Conditions')
            env_output = gr.Markdown('')
            model_output = gr.Markdown('')

    predict_btn.click(
        fn=predict,
        inputs=[city, age, height, gender, smoking, asthma, actual_pefr],
        outputs=[zone_output, summary_output, env_output, model_output],
    )

    gr.Markdown('''
    ---
    Model: Random Forest Regressor | Trained on 5,000 records | Weather: Open-Meteo API
    ''')

if __name__ == '__main__':
    demo.launch()
