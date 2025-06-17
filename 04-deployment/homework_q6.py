#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
import argparse

def run(year, month):
    # --- Параметры ---
    # Путь к модели внутри Docker-контейнера
    model_path = '/app/model.bin' 
    
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

    # --- Расчет и вывод среднего значения ---
    mean_pred = y_pred.mean()
    
    print("-" * 50)
    print(f"✅ Готово! Расчет для данных за {year:04d}-{month:02d}.")
    print(f"Средняя прогнозируемая продолжительность: {mean_pred:.2f}")
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Скрипт для прогнозирования продолжительности поездок на такси.')
    parser.add_argument('--year', type=int, required=True, help='Год для обработки')
    parser.add_argument('--month', type=int, required=True, help='Месяц для обработки')
    args = parser.parse_args()
    run(args.year, args.month)