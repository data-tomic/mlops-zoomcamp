# Используем базовый образ, который уже содержит модель
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Устанавливаем pipenv, чтобы работать с Pipfile.lock
RUN pip install pipenv

# Копируем файлы для установки зависимостей
COPY ["Pipfile", "Pipfile.lock", "./"]

# Устанавливаем зависимости в системный Python, а не в venv
# --deploy гарантирует, что Pipfile.lock не устарел
RUN pipenv install --system --deploy

# Копируем наш скрипт в рабочую директорию /app
COPY homework_q6.py .

# Указываем, какая команда будет запускаться при старте контейнера
ENTRYPOINT ["python", "homework_q6.py"]