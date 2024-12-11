import numpy as np
import pandas as pd

# Функция для нормализации методом Min-Max
def normalize_data_with_min_max(data, min_max_values):
    """
    Нормализует данные на основе заранее вычисленных min и max значений.
    :param data: DataFrame с данными для нормализации.
    :param min_max_values: Словарь с min и max для каждого признака.
    :return: Нормализованный DataFrame.
    """
    for column in min_max_values:
        min_val, max_val = min_max_values[column]
        data[column] = (data[column] - min_val) / (max_val - min_val)
    return data
