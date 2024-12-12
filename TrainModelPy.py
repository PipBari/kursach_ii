import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.dates as mdates

# Функция для нормализации данных (нормализация методом Min-Max)
def normalize_data_with_min_max(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]

# 1. Чтение данных из файла
print("Reading input data...")
input_file = 'data/synthetic_data_with_labels.csv'
data = []
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append({
            'Time': datetime.strptime(row['Time'], '%Y-%m-%d %H:%M:%S'),  # Преобразование времени в datetime
            'Temperature': float(row['Temperature']),  # Температура
            'Pressure': float(row['Pressure']),        # Давление
            'Class': int(row['Class'])                 # Класс (0 = нормальный, 1 = критический)
        })

# 2. Извлечение значений температуры, давления и времени для анализа
temperature_values = [row['Temperature'] for row in data]
pressure_values = [row['Pressure'] for row in data]
time_values = [row['Time'] for row in data]

# 3. Нахождение минимальных и максимальных значений для нормализации
print("Calculating min and max values...")
min_max_values = {
    'Temperature': (min(temperature_values), max(temperature_values)),
    'Pressure': (min(pressure_values), max(pressure_values))
}
print(f"Temperature: Min = {min_max_values['Temperature'][0]}, Max = {min_max_values['Temperature'][1]}")
print(f"Pressure: Min = {min_max_values['Pressure'][0]}, Max = {min_max_values['Pressure'][1]}")

# 4. Построение графиков исходных данных (до нормализации)
plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 1)
plt.scatter(time_values, temperature_values, c='blue', s=10)  # Температура (до нормализации)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Original Temperature')

plt.subplot(2, 2, 2)
plt.scatter(time_values, pressure_values, c='green', s=10)  # Давление (до нормализации)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Original Pressure')

# 5. Нормализация данных (Min-Max нормализация)
print("Normalizing data...")
normalized_temperature = normalize_data_with_min_max(temperature_values, min_max_values['Temperature'][0], min_max_values['Temperature'][1])
normalized_pressure = normalize_data_with_min_max(pressure_values, min_max_values['Pressure'][0], min_max_values['Pressure'][1])

for i, row in enumerate(data):
    row['Temperature'] = normalized_temperature[i]  # Нормализованная температура
    row['Pressure'] = normalized_pressure[i]        # Нормализованное давление

# Построение графиков нормализованных данных
plt.subplot(2, 2, 3)
plt.scatter(time_values, normalized_temperature, c='red', s=10)  # Нормализованная температура
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Normalized Temperature')
plt.title('Normalized Temperature')

plt.subplot(2, 2, 4)
plt.scatter(time_values, normalized_pressure, c='purple', s=10)  # Нормализованное давление
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Normalized Pressure')
plt.title('Normalized Pressure')

plt.tight_layout()
plt.savefig('data/normalization_plots.png')
plt.show()

# 6. Добавление полиномиальных признаков для улучшения модели
print("Adding polynomial features...")
for row in data:
    row['Temp^2'] = row['Temperature'] ** 2           # Квадрат температуры
    row['Press^2'] = row['Pressure'] ** 2             # Квадрат давления
    row['Temp_Press'] = row['Temperature'] * row['Pressure']  # Произведение температуры и давления
    row['Temp^3'] = row['Temperature'] ** 3           # Куб температуры
    row['Press^3'] = row['Pressure'] ** 3             # Куб давления

# 7. Балансировка классов (увеличение данных для класса 0)
print("Balancing classes...")
majority = [row for row in data if row['Class'] == 0]  # Нормальные случаи (Class = 0)
minority = [row for row in data if row['Class'] == 1]  # Критические случаи (Class = 1)

# Увеличение данных меньшинства до размера большинства
majority_upsampled = majority * 2
minority_upsampled = minority * (len(majority_upsampled) // len(minority)) + minority[:len(majority_upsampled) % len(minority)]
balanced_data = majority_upsampled + minority_upsampled

# 8. Разделение на признаки и метки (X - признаки, y - метки)
X = [(row['Temperature'], row['Pressure'], row['Temp^2'], row['Press^2'], row['Temp_Press']) for row in balanced_data]
y = [row['Class'] for row in balanced_data]

# 9. Разделение на обучающую и тестовую выборки (70% - обучение, 30% - тест)
print("Splitting data into training and testing sets...")
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Функция для расчёта метрик качества
print("Defining metrics calculation function...")
def calculate_metrics(y_true, y_pred):
    accuracy = sum(y_pred[i] == y_true[i] for i in range(len(y_true))) / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# 10. Обучение логистической регрессии (реализация с нуля)
print("Training logistic regression model...")
def train_logistic_regression(X_train, y_train, learning_rate, iterations, lambda_reg, class_weights):
    weights = [random.uniform(-0.01, 0.01) for _ in range(len(X_train[0]))]  # Инициализация весов
    bias = random.uniform(-0.01, 0.01)  # Инициализация смещения
    loss_history = []

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    for iteration in range(iterations):
        loss = 0
        for i in range(len(X_train)):
            z = sum(weights[j] * X_train[i][j] for j in range(len(weights))) + bias  # Линейная комбинация
            pred = sigmoid(z)  # Вычисление вероятности
            error = pred - y_train[i]  # Ошибка предсказания
            weight = class_weights[y_train[i]]  # Вес класса

            # Обновление весов и смещения с учётом регуляризации
            for j in range(len(weights)):
                gradient = weight * error * X_train[i][j] + lambda_reg * weights[j]
                weights[j] -= learning_rate * gradient

            bias_gradient = weight * error
            bias -= learning_rate * bias_gradient

            # Расчёт функции потерь
            loss += weight * (-y_train[i] * math.log(pred + 1e-8) - (1 - y_train[i]) * math.log(1 - pred + 1e-8))
        loss /= len(X_train)
        loss_history.append(loss)

        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    return weights, bias, loss_history

# Настройка параметров для обучения
learning_rate = 0.003  # Скорость обучения
iterations = 5000      # Количество итераций
lambda_reg = 0.01      # Коэффициент регуляризации
class_weights = {0: 5.0, 1: 2.0}  # Веса классов для учета дисбаланса

# Запуск обучения модели
weights, bias, loss_history = train_logistic_regression(X_train, y_train, learning_rate, iterations, lambda_reg, class_weights)

# 11. Оценка модели
print("Evaluating model...")
def predict(X, weights, bias):
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    predictions = []
    for x in X:
        z = sum(weights[j] * x[j] for j in range(len(weights))) + bias  # Вычисление линейной комбинации
        predictions.append(sigmoid(z))  # Преобразование в вероятность
    return predictions

# Построение графика классификации
print("Plotting classification results...")
def plot_classification(X_test, predictions):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        [x[0] for x in X_test],
        [x[1] for x in X_test],
        c=predictions, cmap='viridis', s=20
    )
    plt.colorbar(scatter, label='Predicted Class (0: Normal, 1: Critical)')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Normalized Pressure')
    plt.title('Logistic Regression Classification')
    plt.savefig('data/classification_plot.png')
    plt.show()

predictions = predict(X_test, weights, bias)  # Предсказания на тестовых данных
plot_classification(X_test, predictions)

# Бинаризация предсказаний (0 или 1)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
accuracy, precision, recall, f1 = calculate_metrics(y_test, binary_predictions)  # Вычисление метрик

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# 12. Построение графика функции потерь
print("Plotting loss history...")
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), loss_history, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.savefig('data/loss_plot.png')
plt.show()

# 13. Сохранение обученной модели и параметров
print("Saving the trained model...")
with open('data/trained_model_py.pkl', 'wb') as model_file:
    pickle.dump({'weights': weights, 'bias': bias}, model_file)  # Сохранение весов и смещения

with open('data/min_max_values.pkl', 'wb') as min_max_file:
    pickle.dump(min_max_values, min_max_file)  # Сохранение минимальных и максимальных значений для нормализации
