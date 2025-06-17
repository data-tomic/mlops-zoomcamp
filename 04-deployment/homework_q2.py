#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import os

def run():
    """
    Основная функция для выполнения скрипта.
    Загружает модель, считывает данные, делает прогнозы,
    сохраняет результат в Parquet и выводит размер файла.
    """
    # --- Параметры ---
    year = 2023
    month = 3  # Март
    model_path = '/tmp/mlops-zoomcamp/cohorts/2025/04-deployment/homework/model.bin'
    output_file = f'predictions_{year:04d}-{month:02d}.parquet'

    # --- Загрузка модели ---
    print(f"Загрузка модели из файла: {model_path}...")
    try:
        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print(f"Ошибка: Файл модели '{model_path}' не найден.", file=sys.stderr)
        sys.exit(1)
    print("Модель успешно загружена.")

    # --- Подготовка данных ---
    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        """Читает данные из Parquet файла и подготавливает признаки."""
        print(f"Чтение данных из: {filename}...")
        df = pd.read_parquet(filename)
        
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        return df

    input_file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file_url)

    # --- Выполнение прогноза ---
    print("Подготовка признаков и выполнение прогнозов...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # --- Q2: Подготовка и сохранение выходного файла ---
    print("Создание DataFrame с результатами...")
    
    # 1. Создание ride_id
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # 2. Создание DataFrame с результатами
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred

    # 3. Сохранение в Parquet файл
    print(f"Сохранение результатов в файл: {output_file}...")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    
    # --- Расчет и вывод размера файла ---
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024) # Перевод из байт в мегабайты

    print("-" * 50)
    print(f"✅ Готово! Файл '{output_file}' успешно сохранен.")
    print(f"Размер выходного файла: {file_size_mb:.2f} МБ")
    print("-" * 50)


if __name__ == '__main__':
    run()