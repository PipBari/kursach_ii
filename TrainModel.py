import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from sklearn.preprocessing import MinMaxScaler  # Использование встроенной нормализации

# Чтение данных с классами
print("Reading input data...")
input_file = 'data/synthetic_data_with_labels.csv'
data = pd.read_csv(input_file)

# Сохранение копии исходных данных для графиков
original_data = data.copy()

# Вывод минимальных и максимальных значений температуры и давления
print("Calculating min and max values...")
min_temp, max_temp = data['Temperature'].min(), data['Temperature'].max()
min_pressure, max_pressure = data['Pressure'].min(), data['Pressure'].max()
print(f"Temperature: Min = {min_temp}, Max = {max_temp}")
print(f"Pressure: Min = {min_pressure}, Max = {max_pressure}")

# Нормализация данных (исключая класс)
print("Normalizing data...")
scaler = MinMaxScaler()
data[['Temperature', 'Pressure']] = scaler.fit_transform(data[['Temperature', 'Pressure']])

# Балансировка классов через oversampling
print("Balancing classes...")
majority = data[data['Class'] == 0]
minority = data[data['Class'] == 1]

# Увеличение данных для меньшинства
minority_upsampled = resample(minority,
                              replace=True,  # С повторением
                              n_samples=len(majority),  # Уравниваем с количеством в классе 0
                              random_state=42)

# Соединение данных после балансировки
balanced_data = pd.concat([majority, minority_upsampled])

# Построение графиков исходных и нормализованных данных
print("Plotting data...")
plt.figure(figsize=(16, 12))

# Исходные данные (температура)
plt.subplot(2, 2, 1)
plt.scatter(original_data['Time'], original_data['Temperature'], color='blue', s=10)
plt.title('Original Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Исходные данные (давление)
plt.subplot(2, 2, 2)
plt.scatter(original_data['Time'], original_data['Pressure'], color='green', s=10)
plt.title('Original Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')

# Нормализованные данные (температура)
plt.subplot(2, 2, 3)
plt.scatter(data['Time'], data['Temperature'], color='red', s=10)
plt.title('Normalized Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Нормализованные данные (давление)
plt.subplot(2, 2, 4)
plt.scatter(data['Time'], data['Pressure'], color='purple', s=10)
plt.title('Normalized Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')

plt.tight_layout()
plt.savefig('data/normalized_data_plot.png')
plt.show()

# Разделение данных на признаки (X) и метки (y)
X = balanced_data[['Temperature', 'Pressure']]
y = balanced_data['Class']

# Разделение на обучающую и тестовую выборки
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели логистической регрессии
print("Training logistic regression model...")
model = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs', random_state=42, class_weight='balanced', verbose=1)
model.fit(X_train, y_train)

# Предсказание на тестовых данных
print("Making predictions...")
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Визуализация результатов классификации
print("Plotting classification results...")
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Temperature'], X_test['Pressure'], c=y_pred, cmap='coolwarm', s=30, label='Predicted Class')
plt.colorbar(label='Predicted Class (0: Normal, 1: Critical)')
plt.xlabel('Normalized Temperature')
plt.ylabel('Normalized Pressure')
plt.title('Logistic Regression Classification')
plt.legend()
plt.savefig('data/classification_results_balanced.png')
plt.show()

# Сохранение обученной модели и параметров нормализации
print("Saving the trained model and normalization parameters...")
with open('data/trained_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('data/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and normalization parameters saved successfully.")
