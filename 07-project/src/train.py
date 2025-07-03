# 07-project/src/train.py (Версия 2.0 с RandomForest)

import pandas as pd
import mlflow
# --- ИЗМЕНЕНИЕ ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_model():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("TCGA-BRCA-Classification")

    print("1. Загрузка данных...")
    X_df = pd.read_csv("07-project/data/processed/gene_expression_matrix.csv", index_col='gene_id')
    metadata_df = pd.read_csv("07-project/data/metadata.csv", index_col='file_id')

    print("2. Подготовка данных для обучения...")
    X = X_df.T
    y = metadata_df['sample_type'].map({'Primary Tumor': 1, 'Solid Tissue Normal': 0}).reindex(X.index)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("3. Запуск эксперимента в MLflow с RandomForest...")
    with mlflow.start_run():
        # --- ИЗМЕНЕНИЕ: Новые параметры ---
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100, # Количество "деревьев" в лесу
            "max_depth": 10,
            "random_state": 42
        }
        mlflow.log_params(params)
        print("   - Параметры залогированы.")

        # --- ИЗМЕНЕНИЕ: Обучаем новую модель ---
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)
        print("   - Модель обучена.")

        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        print(f"   - Метрики залогированы: Accuracy={accuracy:.4f}, F1-score={f1:.4f}")

        # --- ИЗМЕНЕНИЕ: Логируем новую модель ---
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("   - Модель залогирована как артефакт.")

    print("\nУСПЕХ! Эксперимент завершен.")
    print("Обновите страницу MLflow UI, чтобы увидеть новый запуск.")

if __name__ == "__main__":
    train_model()