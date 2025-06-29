import os
import pandas as pd
from datetime import datetime

# Функция для создания тестовых дат

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# --- Читаем параметры из переменных окружения ---
# Пайплайн передаст сюда значения из UI.
# Если запускаем локально и переменные не заданы, используются значения по умолчанию.

YEAR = int(os.getenv("YEAR", "2023"))
MONTH = int(os.getenv("MONTH", "1"))

print(f"Running integration test for YEAR={YEAR}, MONTH={MONTH}")

# --- Настройки S3 ---
# Скрипт не использует S3_ENDPOINT_URL, поэтому будет работать с AWS по умолчанию
options = {'client_kwargs': {}}

# --- Данные для теста ---
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),   
]
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

# Имя бакета нужно указать явно
S3_BUCKET_NAME = "mlops-zoomcamp-data-tomic-2025"  # <-- УКАЖИТЕ ИМЯ ВАШЕГО БАКЕТА

input_file = f"s3://{S3_BUCKET_NAME}/in/{YEAR:04d}-{MONTH:02d}.parquet"
output_file = f"s3://{S3_BUCKET_NAME}/out/{YEAR:04d}-{MONTH:02d}.parquet"

# --- Шаг 1: Сохраняем тестовые данные в S3 ---
print(f"Step 1: Saving test data to {input_file}")
df_input.to_parquet(input_file, engine='pyarrow', index=False, storage_options=options)

# --- Шаг 2: Запускаем основной скрипт обработки ---
print("\nStep 2: Running the main batch processing script")
# Передаем переменные окружения, чтобы основной скрипт знал, где искать файлы
os.environ['INPUT_FILE_PATTERN'] = f"s3://{S3_BUCKET_NAME}/in/{{year:04d}}-{{month:02d}}.parquet"
os.environ['OUTPUT_FILE_PATTERN'] = f"s3://{S3_BUCKET_NAME}/out/{{year:04d}}-{{month:02d}}.parquet"
# S3_ENDPOINT_URL не задан, поэтому будет использоваться AWS

# Запускаем скрипт
os.system(f"python3 06-best-practices/homework_q1.py {YEAR} {MONTH}")
print("\nMain script finished.")

# --- Шаг 3: Читаем результат и проверяем ---
print("\nStep 3: Reading and checking the result...")
df_result = pd.read_parquet(output_file, storage_options=options)
sum_predicted_durations = df_result['predicted_duration'].sum()

print("\n==========================================")
print(f"RESULT: Sum of predicted durations = {sum_predicted_durations}")
print("==========================================")

# Простая проверка, чтобы пайплайн мог "упасть", если результат неверный
assert abs(sum_predicted_durations - 36.28) < 0.1, "Result is not as expected!"
print("\nIntegration test PASSED!")
