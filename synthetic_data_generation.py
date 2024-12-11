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

time_interval = '60T'  # Интервал времени 2 часа
start_time = pd.Timestamp('2010-01-01')
end_time = pd.Timestamp('2010-01-15')  # Генерация данных за 5 дней
time = pd.date_range(start=start_time, end=end_time, freq=time_interval)

# Генерация нормальной фазы
def generate_normal_phase(num_samples):
    trend = np.linspace(0, 1, num_samples)
    noise = np.random.normal(0, temperature_noise_std, num_samples)
    return base_temperature + trend + noise

# Генерация критической фазы
def generate_critical_phase(num_samples):
    trend = np.linspace(0, 5, num_samples)
    noise = np.random.normal(0, temperature_noise_std * 3, num_samples)
    spikes = np.random.choice([0, 20, -15], size=num_samples, p=[0.9, 0.05, 0.05])
    return max_temperature - 10 + trend + noise + spikes

# Применение суточных колебаний
def apply_diurnal_variation(data, amplitude, period):
    time_indices = np.arange(len(data))
    variation = amplitude * np.sin(2 * np.pi * time_indices / period)
    return data + variation

# Генерация давления
def generate_pressure_based_on_temperature(temperature):
    noise = np.random.normal(0, pressure_noise_std, len(temperature))
    return base_pressure + pressure_scale * (temperature - base_temperature) + noise

# Генерация фаз с балансом 75% на 25%
def generate_phases(num_samples):
    phases = []
    while num_samples > 0:
        phase_type = np.random.choice(['Normal', 'Critical'], p=[0.75, 0.25])
        if phase_type == 'Normal':
            phase_len = np.random.randint(100, 300)
            phases.append(generate_normal_phase(phase_len))
        else:
            phase_len = np.random.randint(50, 150)
            phases.append(generate_critical_phase(phase_len))
        num_samples -= phase_len
    return np.concatenate(phases)[:num_samples]

# Генерация данных
num_samples = len(time)
temperature = generate_phases(num_samples)
temperature = apply_diurnal_variation(temperature, diurnal_variation_amplitude, diurnal_variation_period)
pressure = generate_pressure_based_on_temperature(temperature)

# Сохранение данных
output_folder = 'data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = pd.DataFrame({'Time': time, 'Temperature': temperature, 'Pressure': pressure})
output_file = os.path.join(output_folder, 'synthetic_data.csv')
data.to_csv(output_file, index=False)

# Визуализация
plt.figure(figsize=(16, 8))
plt.plot(data['Time'], data['Temperature'], label='Temperature', color='blue', alpha=0.7)
plt.plot(data['Time'], data['Pressure'], label='Pressure', color='green', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Enhanced Temperature and Pressure Data')
plt.legend()
plt.savefig(os.path.join(output_folder, 'synthetic_data.png'))
plt.show()
