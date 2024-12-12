from flask import Flask, request, render_template
import pickle
import pandas as pd
import math

# Загрузка модели, обученной с библиотеками
with open('data/trained_model.pkl', 'rb') as model_file:
    sklearn_model = pickle.load(model_file)

with open('data/scaler.pkl', 'rb') as scaler_file:
    sklearn_scaler = pickle.load(scaler_file)

# Загрузка модели, обученной без библиотек
with open('data/trained_model_py.pkl', 'rb') as model_file_py:
    custom_model = pickle.load(model_file_py)

# Загрузка минимальных и максимальных значений
with open('data/min_max_values.pkl', 'rb') as min_max_file:
    min_max_values = pickle.load(min_max_file)

# Инициализация Flask приложения
app = Flask(__name__)

# Функция нормализации данных (для модели без библиотек)
def normalize_data_with_min_max(data, min_max_values):
    for column in min_max_values:
        min_val, max_val = min_max_values[column]
        data[column] = (data[column] - min_val) / (max_val - min_val)
    return data

# Функция для предсказания с использованием модели без библиотек
def predict_custom_model(normalized_features, weights, bias):
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))
    z = sum(w * f for w, f in zip(weights, normalized_features)) + bias
    return sigmoid(z)

# Главная страница (модель с библиотеками)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Получение данных из формы
            temperature = float(request.form['temperature'])
            pressure = float(request.form['pressure'])

            # Создание DataFrame и нормализация
            user_data = pd.DataFrame({'Temperature': [temperature], 'Pressure': [pressure]})
            normalized_data = sklearn_scaler.transform(user_data)

            # Прогноз модели
            probability = sklearn_model.predict_proba(normalized_data)[:, 1][0]
            prediction = 1 if probability > 0.5 else 0

            # Жёсткие правила
            if normalized_data[0, 0] > 0.8 or normalized_data[0, 1] > 0.8:
                result = "Critical Situation (Rule-Based)"
            else:
                result = "Critical Situation" if prediction == 1 else "Normal Situation"

            return render_template(
                'index.html',
                result=result,
                probability=round(probability, 2),
                temperature=temperature,
                pressure=pressure
            )
        except Exception as e:
            return render_template('index.html', result=f"Error: {str(e)}")
    return render_template('index.html', result=None)

# Страница для модели, обученной без библиотек
@app.route('/model-data', methods=['GET', 'POST'])
def model_data():
    if request.method == 'POST':
        try:
            # Получение данных из формы
            temperature = float(request.form['temperature'])
            pressure = float(request.form['pressure'])

            # Создание DataFrame для нормализации
            user_data = pd.DataFrame({'Temperature': [temperature], 'Pressure': [pressure]})
            normalized_data = normalize_data_with_min_max(user_data, min_max_values)

            # Извлечение нормализованных данных
            normalized_features = normalized_data.iloc[0].tolist()

            # Добавление полиномиальных признаков
            normalized_features_with_polynomials = normalized_features + [
                normalized_features[0] ** 2,  # Temp^2
                normalized_features[1] ** 2,  # Press^2
                normalized_features[0] * normalized_features[1]  # Temp_Press
            ]

            # Прогноз модели
            probability = predict_custom_model(
                normalized_features_with_polynomials,
                custom_model['weights'],
                custom_model['bias']
            )
            prediction = 1 if probability > 0.5 else 0

            # Жёсткие правила
            if normalized_features[0] > 0.8 or normalized_features[1] > 0.8:
                result = "Critical Situation (Rule-Based)"
            else:
                result = "Critical Situation" if prediction == 1 else "Normal Situation"

            return render_template(
                'model_data.html',
                result=result,
                probability=round(probability, 2),
                temperature=temperature,
                pressure=pressure
            )
        except Exception as e:
            return render_template('model_data.html', result=f"Error: {str(e)}")
    return render_template('model_data.html', result=None)

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
