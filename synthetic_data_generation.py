import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Параметры для генерации синтетических данных
base_temperature = 300  # Базовая температура (нормальное состояние)
max_temperature = 350  # Максимальная температура (критическая точка)
temperature_noise_std = 0.1  # Шум температуры

base_pressure = 100  # Базовое давление
pressure_noise_std = 0.1  # Шум давления
pressure_scale = 0.05  # Масштаб зависимости давления от температуры

diurnal_variation_amplitude = 20  # Суточные колебания температуры
diurnal_variation_period = 24 * 60  # Период суточных колебаний (в минутах)

critical_temperature_threshold = 330  # Порог критической температуры
critical_pressure_threshold = 110  # Порог критического давления

time_interval = '60T'  # Интервал времени 60 минут
start_time = pd.Timestamp('2010-01-01')
end_time = pd.Timestamp('2010-01-15')  # Генерация данных за 15 дней
time = pd.date_range(start=start_time, end=end_time, freq=time_interval)

# Генерация нормальной фазы
def generate_normal_phase(num_samples):
    trend = np.linspace(0, 2, num_samples)
    noise = np.random.normal(0, temperature_noise_std, num_samples)
    return base_temperature + trend + noise

# Генерация критической фазы с аномалиями
def generate_critical_phase(num_samples):
    trend = np.linspace(5, 10, num_samples)
    noise = np.random.normal(0, temperature_noise_std * 3, num_samples)
    spikes = np.random.choice([0, 20, -15, 30, -25], size=num_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05])
    gradual_anomalies = np.linspace(0, 15, num_samples) * np.random.choice([-1, 1], size=num_samples, p=[0.5, 0.5])
    return max_temperature - 10 + trend + noise + spikes + gradual_anomalies

# Применение суточных колебаний
def apply_diurnal_variation(data, amplitude, period):
    time_indices = np.arange(len(data))
    variation = amplitude * np.sin(2 * np.pi * time_indices / period)
    return data + variation

# Генерация давления с нелинейными зависимостями
def generate_pressure_based_on_temperature(temperature):
    noise = np.random.normal(0, pressure_noise_std, len(temperature))
    non_linear_effect = np.sin(temperature / 50) * 5  # Нелинейный эффект
    return base_pressure + pressure_scale * (temperature - base_temperature) + noise + non_linear_effect

# Генерация фаз с классами (Normal=0, Critical=1)
def generate_phases_with_labels(num_samples):
    phases = []
    labels = []
    while num_samples > 0:
        phase_type = np.random.choice(['Normal', 'Critical'], p=[0.7, 0.3])
        if phase_type == 'Normal':
            phase_len = np.random.randint(100, 300)
            phases.append(generate_normal_phase(phase_len))
            labels.extend([0] * phase_len)
        else:
            phase_len = np.random.randint(50, 150)
            phases.append(generate_critical_phase(phase_len))
            labels.extend([1] * phase_len)
        num_samples -= phase_len
    return np.concatenate(phases)[:num_samples], labels[:num_samples]

# Генерация данных
num_samples = len(time)
temperature, labels = generate_phases_with_labels(num_samples)
temperature = apply_diurnal_variation(temperature, diurnal_variation_amplitude, diurnal_variation_period)
pressure = generate_pressure_based_on_temperature(temperature)

# Обновление классов на основе критических порогов
labels = [1 if temp > critical_temperature_threshold or pres > critical_pressure_threshold else 0
          for temp, pres in zip(temperature, pressure)]

# Сохранение данных
output_folder = 'data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = pd.DataFrame({
    'Time': time,
    'Temperature': temperature,
    'Pressure': pressure,
    'Class': labels
})
output_file = os.path.join(output_folder, 'synthetic_data_with_labels.csv')
data.to_csv(output_file, index=False)

# Визуализация
plt.figure(figsize=(16, 8))
plt.plot(data['Time'], data['Temperature'], label='Temperature', color='blue', alpha=0.7)
plt.plot(data['Time'], data['Pressure'], label='Pressure', color='green', alpha=0.7)
plt.fill_between(data['Time'], 0, 1, where=(data['Class'] == 1), color='red', alpha=0.1, transform=plt.gca().get_xaxis_transform(), label='Critical Phase')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Enhanced Temperature and Pressure Data')
plt.legend()
plt.savefig(os.path.join(output_folder, 'synthetic_data_with_labels.png'))
plt.show()
