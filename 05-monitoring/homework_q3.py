# ==========================================================
#  ФИНАЛЬНЫЙ СКРИПТ, КОТОРЫЙ ПРОБУЕТ ВСЕ СИНТАКСИСЫ
# ==========================================================
import datetime
import pandas as pd
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

# --- Этап 1: Динамический импорт ---
SYNTAX_MODE = None
try:
    # Попытка №1: Современный синтаксис (v0.4+)
    from evidently.report import Report
    from evidently.column_mapping import ColumnMapping
    from evidently.metrics import ColumnQuantileMetric

    SYNTAX_MODE = "MODERN"
    logging.info("Успешно импортирован СОВРЕМЕННЫЙ синтаксис evidently.")
except ImportError:
    logging.warning("Современный синтаксис не удался. Пробуем старый...")
    try:
        # Попытка №2: Старый синтаксис (v0.2-v0.3)
        from evidently import Report, Dataset, DataDefinition
        from evidently.metrics import ColumnQuantileMetric

        SYNTAX_MODE = "LEGACY"
        logging.info("Успешно импортирован СТАРЫЙ синтаксис evidently.")
    except ImportError as e:
        logging.error(
            "!!! КРИТИЧЕСКАЯ ОШИБКА: Ни один из известных синтаксисов не сработал."
        )
        logging.error(f"Финальная ошибка: {e}")
        sys.exit()

# --- Этап 2: Загрузка данных ---
logging.info("Загрузка данных по URL...")
try:
    raw_data = pd.read_parquet(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-03.parquet"
    )
except Exception as e:
    logging.error(f"Не удалось загрузить файл по URL: {e}")
    sys.exit()
logging.info("Данные загружены.")

# --- Этап 3: Подготовка отчета в зависимости от синтаксиса ---
if SYNTAX_MODE == "MODERN":
    report = Report(
        metrics=[ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)]
    )
    column_mapping = ColumnMapping(numerical_features=["fare_amount"])
elif SYNTAX_MODE == "LEGACY":
    report = Report(
        metrics=[ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)]
    )
    data_definition = DataDefinition(numerical_columns=["fare_amount"])

# --- Этап 4: Основной цикл расчета ---
begin = datetime.datetime(2024, 3, 1)
daily_metrics = []

logging.info("Начало расчета метрик по дням...")
for i in range(31):
    current_date = begin + datetime.timedelta(i)
    next_date = begin + datetime.timedelta(i + 1)

    daily_data = raw_data[
        (pd.to_datetime(raw_data.lpep_pickup_datetime) >= current_date)
        & (pd.to_datetime(raw_data.lpep_pickup_datetime) < next_date)
    ].copy()

    if daily_data.empty:
        continue

    # --- Запускаем отчет, используя правильный синтаксис ---
    if SYNTAX_MODE == "MODERN":
        report.run(
            reference_data=None, current_data=daily_data, column_mapping=column_mapping
        )
        result = report.as_dict()
        value = result["metrics"][0]["result"]["current"]["value"]
    elif SYNTAX_MODE == "LEGACY":
        daily_dataset = Dataset.from_pandas(daily_data, data_definition)
        report.run(reference_data=None, current_data=daily_dataset)
        result = report.dict()
        value = result["metrics"][0]["result"]["value"]

    daily_metrics.append(value)
    logging.info(f"День {current_date.date()}: Квантиль fare_amount (0.5) = {value}")

# --- Этап 5: Итоговый результат ---
if daily_metrics:
    max_quantile = max(daily_metrics)
    logging.info("=" * 50)
    print(
        f"\n\nОТВЕТ НА Q3: МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ КВАНТИЛЯ (0.5) ДЛЯ fare_amount: {max_quantile}\n\n"  # noqa: E501
    )
else:
    logging.error("Не удалось рассчитать ни одной метрики.")
