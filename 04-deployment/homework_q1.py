#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

def run():
    """
    Основная функция для выполнения скрипта.
    Загружает модель, считывает данные за указанный год и месяц,
    делает прогнозы и вычисляет стандартное отклонение.
    """
    # --- Параметры ---
    year = 2023
    month = 3  # Март
    model_path = '/tmp/mlops-zoomcamp/cohorts/2025/04-deployment/homework/model.bin'

    # --- Загрузка модели ---
    print(f"Загрузка модели из файла: {model_path}...")
    try:
        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print(f"Ошибка: Файл модели '{model_path}' не найден.", file=sys.stderr)
        print("Пожалуйста, убедитесь, что файл находится в том же каталоге, что и скрипт.", file=sys.stderr)
        sys.exit(1)
    print("Модель успешно загружена.")

    # --- Подготовка данных ---
    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        """Читает данные из Parquet файла и подготавливает признаки."""
        print(f"Чтение данных из: {filename}...")
        df = pd.read_parquet(filename)
        
        # Вычисление продолжительности (для справки, в этой задаче не используется)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        # Фильтрация выбросов (стандартная практика из курса)
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        # Преобразование категориальных признаков в строки
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    # URL файла данных за Март 2023
    input_file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    
    # Чтение и подготовка данных
    df = read_data(input_file_url)

    # --- Выполнение прогноза ---
    print("Подготовка признаков для прогнозирования...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    print("Выполнение прогнозов...")
    y_pred = model.predict(X_val)

    # --- Расчет и вывод результата ---
    std_dev = y_pred.std()
    
    print("-" * 50)
    print(f"✅ Готово! Расчет для данных за {year:04d}-{month:02d}.")
    print(f"Стандартное отклонение прогнозируемой продолжительности: {std_dev:.2f}")
    print("-" * 50)

if __name__ == '__main__':
    run()