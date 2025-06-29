import pandas as pd

# Предполагается, что вы вынесли логику из ноутбука в скрипт, например, `homework.py`
# Если ваш код все еще в ноутбуке, этот шаг потребует небольшого рефакторинга.
# from .. import homework  # Пример импорта


def test_data_preparation():
    """
    Простой дымовой тест: проверяем, что данные читаются и обрабатываются
    без ошибок и имеют ожидаемую структуру.
    """
    # Этот код можно адаптировать под вашу функцию
    # Здесь мы создаем фейковый DataFrame для теста
    data = [
        {
            'tpep_pickup_datetime': '2023-01-01 00:00:00',
            'tpep_dropoff_datetime': '2023-01-01 00:10:00',
            'PULocationID': 1, 'DOLocationID': 2
        },
        {
            'tpep_pickup_datetime': '2023-01-01 00:05:00',
            'tpep_dropoff_datetime': '2023-01-01 00:30:00',
            'PULocationID': 3, 'DOLocationID': 4
        },
    ]
    df = pd.DataFrame(data)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Симулируем вашу логику
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    assert len(df) == 2  # Обе поездки должны остаться
    assert 'duration' in df.columnsv
