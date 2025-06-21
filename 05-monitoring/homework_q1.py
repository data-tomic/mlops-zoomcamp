import pandas as pd

# URL файла с данными за март 2024
url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet'

# Чтение данных из Parquet файла
# Убедитесь, что у вас установлена библиотека pyarrow или fastparquet
# pip install pyarrow
df = pd.read_parquet(url)

# Вывод размера DataFrame (количество строк, количество столбцов)
print(f"Форма загруженных данных: {df.shape}")

# Количество строк для ответа на Q1
num_rows = df.shape[0]
print(f"Количество строк: {num_rows}")