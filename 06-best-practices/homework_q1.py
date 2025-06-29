# homework_q1.py

import os
import sys
import pickle
import pandas as pd

# НОВАЯ функция, которая содержит ТОЛЬКО логику трансформации
def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# ИЗМЕНЕННАЯ функция для чтения
def read_data(filename, categorical):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL,
                # Добавляем ключи, если они есть в окружении
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'verify': False
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    
    return prepare_data(df, categorical)

# НОВАЯ функция для сохранения (понадобится в Q6)
def save_data(df, filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL,
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'verify': False
            }
        }
        df.to_parquet(filename, engine='pyarrow', index=False, storage_options=options)
    else:
        df.to_parquet(filename, engine='pyarrow', index=False)


# ИЗМЕНЕННАЯ функция main
def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    
    df = read_data(input_file, categorical=categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted sum of duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # Используем нашу новую функцию для сохранения
    save_data(df_result, output_file)

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year=year, month=month)