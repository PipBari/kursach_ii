import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import csv
from sklearn.metrics import precision_score, recall_score, f1_score


# Функция для нормализации данных
def normalize_data_with_min_max(data, min_val, max_val):
    return [(x - min_val) / (max_val - min_val) for x in data]


# Чтение исходных данных
print("Reading input data...")
input_file = 'data/synthetic_data.csv'
data = []
with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append({
            'Time': row['Time'],
            'Temperature': float(row['Temperature']),
            'Pressure': float(row['Pressure'])
        })

# Преобразование данных для нормализации
temperature_values = [row['Temperature'] for row in data]
pressure_values = [row['Pressure'] for row in data]

# Нахождение минимальных и максимальных значений
print("Calculating min and max for normalization...")
min_max_values = {
    'Temperature': (min(temperature_values), max(temperature_values)),
    'Pressure': (min(pressure_values), max(pressure_values))
}

# Нормализация данных
print("Normalizing data...")
for row in data:
    row['Temperature'] = (row['Temperature'] - min_max_values['Temperature'][0]) / (
            min_max_values['Temperature'][1] - min_max_values['Temperature'][0])
    row['Pressure'] = (row['Pressure'] - min_max_values['Pressure'][0]) / (
            min_max_values['Pressure'][1] - min_max_values['Pressure'][0])

# Проверка нормализации
print(
    f"Temperature range after normalization: min={min([row['Temperature'] for row in data])}, max={max([row['Temperature'] for row in data])}")
print(
    f"Pressure range after normalization: min={min([row['Pressure'] for row in data])}, max={max([row['Pressure'] for row in data])}")

# Построение графиков нормализованных данных
plt.figure(figsize=(16, 6))

# График температуры
plt.subplot(1, 2, 1)
plt.scatter([row['Time'] for row in data], [row['Temperature'] for row in data], color='blue', s=10)
plt.title('Normalized Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')

# График давления
plt.subplot(1, 2, 2)
plt.scatter([row['Time'] for row in data], [row['Pressure'] for row in data], color='green', s=10)
plt.title('Normalized Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')

plt.tight_layout()
plt.savefig('data/normalized_data_plot.png')
plt.show()

# Добавление новых признаков
print("Adding polynomial features...")
for row in data:
    row['Temp^2'] = row['Temperature'] ** 2
    row['Press^2'] = row['Pressure'] ** 2
    row['Temp_Press'] = row['Temperature'] * row['Pressure']
    row['Temp^3'] = row['Temperature'] ** 3
    row['Press^3'] = row['Pressure'] ** 3
    row['Temp^4'] = row['Temperature'] ** 4
    row['Press^4'] = row['Pressure'] ** 4
    row['Temp^2_Press'] = (row['Temperature'] ** 2) * row['Pressure']
    row['Temp_Press^2'] = row['Temperature'] * (row['Pressure'] ** 2)

# Определение меток классов
print("Defining class labels...")
for row in data:
    row['Class'] = 1 if row['Temperature'] > 0.8 else 0

# Проверка распределения классов до балансировки
print("Class distribution before balancing:")
print(f"Class 0: {sum(row['Class'] == 0 for row in data)}")
print(f"Class 1: {sum(row['Class'] == 1 for row in data)}")

# Балансировка классов
print("Balancing classes...")
majority = [row for row in data if row['Class'] == 0]
minority = [row for row in data if row['Class'] == 1]
minority_upsampled = minority * (len(majority) // len(minority)) + minority[:len(majority) % len(minority)]
balanced_data = majority + minority_upsampled

# Проверка распределения классов после балансировки
print("Class distribution after balancing:")
print(f"Class 0: {len([row for row in balanced_data if row['Class'] == 0])}")
print(f"Class 1: {len([row for row in balanced_data if row['Class'] == 1])}")

# Разделение на признаки и метки
X = [(row['Temperature'], row['Pressure'], row['Temp^2'], row['Press^2'], row['Temp_Press'], row['Temp^3'],
      row['Press^3'], row['Temp^4'], row['Press^4'], row['Temp^2_Press'], row['Temp_Press^2']) for row in balanced_data]
y = [row['Class'] for row in balanced_data]

# Разделение на обучающую и тестовую выборки
print("Splitting data into training and testing sets...")
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Логистическая регрессия с адаптивным градиентным спуском
print("Training logistic regression model...")
weights = [random.uniform(-0.01, 0.01) for _ in range(len(X_train[0]))]
bias = random.uniform(-0.01, 0.01)
learning_rate = 0.003
iterations = 15000  # Увеличено количество итераций
lambda_reg = 0.01  # Ещё снижена регуляризация
class_weights = {0: 2.5, 1: 1.0}  # Ещё увеличен вес класса 0


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def update_adaptive_gradient(value, gradient, gradient_square, learning_rate):
    gradient_square += gradient ** 2
    adjusted_lr = learning_rate / (1e-8 + math.sqrt(gradient_square))
    return value - adjusted_lr * gradient, gradient_square


gradient_squares = [0.0] * len(weights)
bias_square = 0.0
loss_history = []


def plot_decision_boundary_2d(X, y, weights, bias):
    # Построение сетки значений для отображения поверхности
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Формирование полиномиальных признаков для сетки
    temp_sq = xx.ravel() ** 2
    press_sq = yy.ravel() ** 2
    temp_press = xx.ravel() * yy.ravel()
    temp_cub = xx.ravel() ** 3
    press_cub = yy.ravel() ** 3
    temp_quart = xx.ravel() ** 4
    press_quart = yy.ravel() ** 4
    temp_sq_press = (xx.ravel() ** 2) * yy.ravel()
    temp_press_sq = xx.ravel() * (yy.ravel() ** 2)

    grid_features = np.c_[
        xx.ravel(), yy.ravel(), temp_sq, press_sq, temp_press, temp_cub, press_cub, temp_quart, press_quart, temp_sq_press, temp_press_sq]

    # Вычисление предсказаний для сетки
    z = [sigmoid(sum(weights[j] * grid_features[i][j] for j in range(len(weights))) + bias) for i in
         range(len(grid_features))]
    z = np.array(z).reshape(xx.shape)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.6)
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(
        [x[0] for x in X],
        [x[1] for x in X],
        c=colors,
        edgecolor="k",
        s=30
    )
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Normalized Pressure')
    plt.title('Logistic Regression Decision Boundary (2D)')
    plt.show()


for iteration in range(iterations):
    loss = 0
    for i in range(len(X_train)):
        z = sum(weights[j] * X_train[i][j] for j in range(len(weights))) + bias
        pred = sigmoid(z)
        error = pred - y_train[i]
        weight = class_weights[y_train[i]]

        for j in range(len(weights)):
            gradient = weight * error * X_train[i][j] + lambda_reg * weights[j]
            weights[j], gradient_squares[j] = update_adaptive_gradient(weights[j], gradient, gradient_squares[j],
                                                                       learning_rate)

        bias_gradient = weight * error
        bias, bias_square = update_adaptive_gradient(bias, bias_gradient, bias_square, learning_rate)

        loss += weight * (-y_train[i] * math.log(pred + 1e-8) - (1 - y_train[i]) * math.log(1 - pred + 1e-8))
    loss /= len(X_train)
    loss_history.append(loss)

    if iteration % 500 == 0:
        print(f"Iteration {iteration}, Loss: {loss:.4f}")
        print(f"Current Weights: {weights}, Bias: {bias}")

# Оценка модели
print("Making predictions...")


def predict(X):
    return [1 if sigmoid(sum(weights[j] * x[j] for j in range(len(weights))) + bias) > 0.5 else 0 for x in X]


y_pred = predict(X_test)

accuracy = sum(y_pred[i] == y_test[i] for i in range(len(y_test))) / len(y_test)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

conf_matrix = {
    (0, 0): sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0),
    (0, 1): sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1),
    (1, 0): sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0),
    (1, 1): sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1),
}

print("\nConfusion Matrix:")
print(conf_matrix)

# Сохранение модели
print("Saving the trained model...")
with open('data/trained_model.pkl', 'wb') as model_file:
    pickle.dump({'weights': weights, 'bias': bias}, model_file)

with open('data/min_max_values.pkl', 'wb') as min_max_file:
    pickle.dump(min_max_values, min_max_file)

# Графики
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), loss_history, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.savefig('data/loss_plot.png')
plt.show()

plot_decision_boundary_2d(X_test, y_test, weights, bias)

print("Model training and evaluation completed.")
