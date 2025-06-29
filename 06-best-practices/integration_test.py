# integration_test.py

import os
import pandas as pd
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


# --- Настройки для MinIO ---
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://s3.k8s.dgoi.ru")
options = {
    "client_kwargs": {
        "endpoint_url": S3_ENDPOINT_URL,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "verify": False,
    }
}

# --- Данные для теста ---
data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]
columns = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]
df_input = pd.DataFrame(data, columns=columns)

# --- Пути и параметры ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ВОТ ПРАВИЛЬНОЕ МЕСТО ДЛЯ ОПРЕДЕЛЕНИЯ year и month
year = 2023
month = 1
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# УДАЛИТЕ ЭТИ СТРОКИ ИЗ ВАШЕГО integration_test.py
# year = int(sys.argv[1])
# month = int(sys.argv[2])

input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
output_file = f"s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

# Шаг 1: Сохраняем тестовые данные в S3
print(f"Шаг 1: Сохранение тестовых данных в {input_file}")
df_input.to_parquet(
    input_file, engine="pyarrow", compression=None, index=False, storage_options=options
)

# Шаг 2: Запускаем основной скрипт обработки
print("\nШаг 2: Запуск основного скрипта homework_q1.py")
os.environ["INPUT_FILE_PATTERN"] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
os.environ["OUTPUT_FILE_PATTERN"] = (
    "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
)
os.environ["S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
os.system(f"python3 homework_q1.py {year} {month}")
print("\nОсновной скрипт завершил работу.")

# Шаг 3: Читаем результат и проверяем
print("\nШаг 3: Чтение и проверка результата...")
df_result = pd.read_parquet(output_file, storage_options=options)
sum_predicted_durations = df_result["predicted_duration"].sum()

print("\n==========================================")
print(f"ИТОГ: Сумма предсказанных длительностей = {sum_predicted_durations}")
print("==========================================")
