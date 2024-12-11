from flask import Flask, request, render_template
import pickle
import pandas as pd

# Загрузка модели и параметров нормализации
with open('data/trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('data/min_max_values.pkl', 'rb') as min_max_file:
    min_max_values = pickle.load(min_max_file)

# Инициализация Flask приложения
app = Flask(__name__)

# Главная страница с формой ввода параметров
@app.route('/')
def index():
    return render_template('index.html', result=None)

# Функция нормализации данных
def normalize_data_with_min_max(data, min_max_values):
    for column in min_max_values:
        min_val, max_val = min_max_values[column]
        data[column] = (data[column] - min_val) / (max_val - min_val)
    return data

# Обработка ввода данных и прогноз
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        temperature = float(request.form['temperature'])
        pressure = float(request.form['pressure'])

        # Создание DataFrame и нормализация
        user_data = pd.DataFrame({'Temperature': [temperature], 'Pressure': [pressure]})
        normalized_data = normalize_data_with_min_max(user_data, min_max_values)

        # Прогноз модели
        probability = model.predict_proba(normalized_data[['Temperature', 'Pressure']])[:, 1][0]
        threshold = 0.4
        prediction = 1 if probability > threshold else 0

        # Жёсткие правила
        if normalized_data['Temperature'][0] > 0.8 or normalized_data['Pressure'][0] > 0.8:
            result = "Critical Situation (Rule-Based)"
        else:
            result = "Critical Situation" if prediction == 1 else "Normal Situation"

        return render_template('index.html', result=result, probability=round(probability, 2))
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
