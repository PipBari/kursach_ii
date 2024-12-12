import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.dates as mdates

# Функция для нормализации данных
def normalize_data_with_min_max(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]

# Чтение данных из файла
print("Reading input data...")
input_file = 'data/synthetic_data_with_labels.csv'
data = []
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append({
            'Time': datetime.strptime(row['Time'], '%Y-%m-%d %H:%M:%S'),
            'Temperature': float(row['Temperature']),
            'Pressure': float(row['Pressure']),
            'Class': int(row['Class'])
        })

# Извлечение значений температуры, давления и времени
temperature_values = [row['Temperature'] for row in data]
pressure_values = [row['Pressure'] for row in data]
time_values = [row['Time'] for row in data]

# Нахождение минимальных и максимальных значений
print("Calculating min and max values...")
min_max_values = {
    'Temperature': (min(temperature_values), max(temperature_values)),
    'Pressure': (min(pressure_values), max(pressure_values))
}
print(f"Temperature: Min = {min_max_values['Temperature'][0]}, Max = {min_max_values['Temperature'][1]}")
print(f"Pressure: Min = {min_max_values['Pressure'][0]}, Max = {min_max_values['Pressure'][1]}")

# Построение графиков до нормализации
plt.figure(figsize=(14, 12))
plt.subplot(2, 2, 1)
plt.scatter(time_values, temperature_values, c='blue', s=10)  # Установлен размер точек
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Original Temperature')

plt.subplot(2, 2, 2)
plt.scatter(time_values, pressure_values, c='green', s=10)  # Установлен размер точек
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Original Pressure')

# Нормализация данных
print("Normalizing data...")
normalized_temperature = normalize_data_with_min_max(temperature_values, min_max_values['Temperature'][0], min_max_values['Temperature'][1])
normalized_pressure = normalize_data_with_min_max(pressure_values, min_max_values['Pressure'][0], min_max_values['Pressure'][1])

for i, row in enumerate(data):
    row['Temperature'] = normalized_temperature[i]
    row['Pressure'] = normalized_pressure[i]

# Построение графиков после нормализации
plt.subplot(2, 2, 3)
plt.scatter(time_values, normalized_temperature, c='red', s=10)  # Установлен размер точек
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Normalized Temperature')
plt.title('Normalized Temperature')

plt.subplot(2, 2, 4)
plt.scatter(time_values, normalized_pressure, c='purple', s=10)  # Установлен размер точек
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel('Time')
plt.ylabel('Normalized Pressure')
plt.title('Normalized Pressure')

plt.tight_layout()
plt.savefig('data/normalization_plots.png')
plt.show()

# Добавление полиномиальных признаков
print("Adding polynomial features...")
for row in data:
    row['Temp^2'] = row['Temperature'] ** 2
    row['Press^2'] = row['Pressure'] ** 2
    row['Temp_Press'] = row['Temperature'] * row['Pressure']
    row['Temp^3'] = row['Temperature'] ** 3
    row['Press^3'] = row['Pressure'] ** 3

# Балансировка классов с увеличением данных для класса 0
print("Balancing classes...")
majority = [row for row in data if row['Class'] == 0]
minority = [row for row in data if row['Class'] == 1]
majority_upsampled = majority * 2  # Увеличиваем количество нормальных данных
minority_upsampled = minority * (len(majority_upsampled) // len(minority)) + minority[:len(majority_upsampled) % len(minority)]
balanced_data = majority_upsampled + minority_upsampled

# Разделение на признаки и метки
X = [(row['Temperature'], row['Pressure'], row['Temp^2'], row['Press^2'], row['Temp_Press']) for row in balanced_data]
y = [row['Class'] for row in balanced_data]

# Разделение на обучающую и тестовую выборки
print("Splitting data into training and testing sets...")
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Функция для расчета метрик
def calculate_metrics(y_true, y_pred):
    accuracy = sum(y_pred[i] == y_true[i] for i in range(len(y_true))) / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# Логистическая регрессия
print("Training logistic regression model...")
def train_logistic_regression(X_train, y_train, learning_rate, iterations, lambda_reg, class_weights):
    weights = [random.uniform(-0.01, 0.01) for _ in range(len(X_train[0]))]
    bias = random.uniform(-0.01, 0.01)
    loss_history = []

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    for iteration in range(iterations):
        loss = 0
        for i in range(len(X_train)):
            z = sum(weights[j] * X_train[i][j] for j in range(len(weights))) + bias
            pred = sigmoid(z)
            error = pred - y_train[i]
            weight = class_weights[y_train[i]]

            for j in range(len(weights)):
                gradient = weight * error * X_train[i][j] + lambda_reg * weights[j]
                weights[j] -= learning_rate * gradient

            bias_gradient = weight * error
            bias -= learning_rate * bias_gradient

            loss += weight * (-y_train[i] * math.log(pred + 1e-8) - (1 - y_train[i]) * math.log(1 - pred + 1e-8))
        loss /= len(X_train)
        loss_history.append(loss)

        if iteration % 500 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    return weights, bias, loss_history

# Настройка параметров модели
learning_rate = 0.003
iterations = 5000
lambda_reg = 0.01
class_weights = {0: 5.0, 1: 2.0}  # Увеличен вес класса 1 для повышения Recall

weights, bias, loss_history = train_logistic_regression(X_train, y_train, learning_rate, iterations, lambda_reg, class_weights)

# Оценка модели
print("Generating classification plot...")
def predict(X, weights, bias):
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    predictions = []
    for x in X:
        z = sum(weights[j] * x[j] for j in range(len(weights))) + bias
        predictions.append(sigmoid(z))
    return predictions

# Построение графика классификации
def plot_classification(X_test, predictions):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        [x[0] for x in X_test],
        [x[1] for x in X_test],
        c=predictions, cmap='viridis', s=20  # Установлен размер точек, убрана обводка
    )
    plt.colorbar(scatter, label='Predicted Class (0: Normal, 1: Critical)')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Normalized Pressure')
    plt.title('Logistic Regression Classification')
    plt.show()

predictions = predict(X_test, weights, bias)
plot_classification(X_test, predictions)

binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
accuracy, precision, recall, f1 = calculate_metrics(y_test, binary_predictions)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Построение графика ошибки
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), loss_history, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.savefig('data/loss_plot.png')
plt.show()

# Сохранение модели
print("Saving the trained model...")
with open('data/trained_model_py.pkl', 'wb') as model_file:
    pickle.dump({'weights': weights, 'bias': bias}, model_file)

with open('data/min_max_values.pkl', 'wb') as min_max_file:
    pickle.dump(min_max_values, min_max_file)